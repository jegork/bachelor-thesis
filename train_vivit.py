import os
import io
import imageio as iio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from vivit import *

SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)


# DATA
BATCH_SIZE = 50
INPUT_SHAPE = (100, 64, 64, 3)
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
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

ATTENTION_DROPOUT=0.3
SOFTMAX_DROPOUT=0.3

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
def decode_video(serialized_image, feature): 
    img = tf.io.decode_jpeg(serialized_image, channels=INPUT_SHAPE[-1])
    return tf.image.resize_with_pad(
            img,
            target_height = INPUT_SHAPE[1],
            target_width = INPUT_SHAPE[1],
            method = 'bilinear')

def resample_video(video, label):
    mask = [True if i in range(0, video.shape[0], EVERY_N_FRAME) else False for i in range(video.shape[0])]
    video = tf.boolean_mask(video, mask, axis=0)
    video_len = tf.shape(video)[0].numpy()
    
    if video_len < INPUT_SHAPE[0]:
        paddings = tf.constant([[0, INPUT_SHAPE[0] - video_len], [0, 0], [0, 0], [0, 0]])
        video = tf.pad(video, paddings, 'CONSTANT')
    elif video_len > INPUT_SHAPE[0]:
        video = video[:INPUT_SHAPE[0]]
    
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

    _resample = lambda x: tf.py_function(resample_video, [x['video'], x['label']], Tout=(tf.TensorSpec(INPUT_SHAPE, tf.float32), tf.int64))
    return dataset.map(_resample).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == '__main__':
    import sys
    from dotenv import dotenv_values

    if os.path.exists('.env'):
        env = dotenv_values(".env")  
    
    trainloader = prepare_dataloader("train")
    testloader = prepare_dataloader("test")

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
        NUM_CLASSES
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
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
                'softmax_dropout': SOFTMAX_DROPOUT
            }

            callbacks.append(NeptuneCallback(run=run, base_namespace=""))
        else:
            raise Exception('Please provide .env file with NEPTUNE_API_KEY')

    # Train the model.
    h = model.fit(trainloader, epochs=EPOCHS, validation_data=testloader, callbacks=callbacks)
