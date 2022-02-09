import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from vivit import *
from tf_video import VideoRandomFlip, VideoRandomContrast, VideoRandomZoom, VideoRandomRotation, VideoRandomCrop

SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)


# DATA
BATCH_SIZE = 60
INPUT_SHAPE = (50, 128, 128, 3)
NUM_CLASSES = 101
EVERY_N_FRAME = 3

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6

ATTENTION_DROPOUT=0.3
SOFTMAX_DROPOUT=0.3
LABEL_SMOOTHING=0.3

tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_video(serialized_image, features): 
    img = tf.io.decode_jpeg(serialized_image, channels=INPUT_SHAPE[-1])
    img = tf.cast(img, tf.float32)/255
    
    return tf.image.resize_with_pad(
            img,
            target_height = INPUT_SHAPE[1],
            target_width = INPUT_SHAPE[1],
            method = 'bilinear')

def preprocess_data(video, label):
    input_length = tf.shape(video)[0]
            
    mask_range = tf.range(0, input_length, EVERY_N_FRAME)
    video = tf.gather(video, mask_range, axis=0)
    
    video_length = tf.shape(video)[0]

    def cut():
        return video[:INPUT_SHAPE[0]]
    
    def pad():
        pad_n = tf.subtract(INPUT_SHAPE[0], video_length)
        tf.ensure_shape(pad_n, ())
        paddings = [[0, pad_n], [0, 0], [0, 0], [0, 0]]
        return tf.pad(video, paddings, 'CONSTANT')
    
    video = tf.cond(tf.math.less(video_length, INPUT_SHAPE[0]), pad, cut)

    label = tf.one_hot(label, 101)

    return video, label

def prepare_dataloader(
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""

    dataset = tfds.load('ucf101/ucf101_1', 
            split=loader_type, 
            decoders={'video': decode_video()},
            download_and_prepare_kwargs={'download_config': tfds.download.DownloadConfig(verify_ssl=False)})

    return dataset.map(lambda x: preprocess_data(x['video'], x['label']))\
        .shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == '__main__':
    import sys
    from dotenv import dotenv_values

    if os.path.exists('.env'):
        env = dotenv_values(".env")  
    
    trainloader = prepare_dataloader("train")
    testloader = prepare_dataloader("test")

    augmentation = tf.keras.models.Sequential([
            VideoRandomFlip('horizontal_and_vertical'), 
            VideoRandomContrast(0.3), 
            VideoRandomZoom((-0.25, -0.5), (-0.25, -0.5)),
            VideoRandomRotation(45)
        ], name='preprocessing')

    # Initialize model
    model = create_vivit_classifier(
        TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
            ),
        PositionalEncoder(embed_dim=PROJECTION_DIM),
        INPUT_SHAPE,
        NUM_LAYERS,
        NUM_HEADS,
        PROJECTION_DIM,
        ATTENTION_DROPOUT,
        SOFTMAX_DROPOUT,
        LAYER_NORM_EPS,
        NUM_CLASSES,
        augmentation
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
            optimizer=optimizer,
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ],
            )
    
    callbacks = []
    
    if '--neptune' in sys.argv:
        if env is not None and 'NEPTUNE_API_KEY' in env:
            import neptune.new as neptune
            from neptune.new.integrations.tensorflow_keras import NeptuneCallback

            run = neptune.init(
                project="jegork/thesis",
                api_token=env['NEPTUNE_API_KEY'],
            )  # your credentials


            run['parameters'] = {
                'batch_size': BATCH_SIZE, 
                'input_shape': INPUT_SHAPE,
                'every_n_frame': EVERY_N_FRAME,
                'lr': LEARNING_RATE, 
                'decay': WEIGHT_DECAY, 
                'epochs': EPOCHS, 
                'patch_size': PATCH_SIZE,
                'layer_norm_eps': LAYER_NORM_EPS,
                'projection_dim': PROJECTION_DIM,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'attention_dropout': ATTENTION_DROPOUT,
                'softmax_dropout': SOFTMAX_DROPOUT,
                'label_smoothing': LABEL_SMOOTHING,
            }

            callbacks.append(NeptuneCallback(run=run, base_namespace=""))
        else:
            raise Exception('Please provide .env file with NEPTUNE_API_KEY')

    # Train the model.
    h = model.fit(trainloader, epochs=EPOCHS, validation_data=testloader, callbacks=callbacks)
