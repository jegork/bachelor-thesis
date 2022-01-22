import os
import io
import imageio as iio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from functools import partial


SEED = 42
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
keras.utils.set_random_seed(SEED)


# DATA
BATCH_SIZE = 100
INPUT_SHAPE = (50, 64, 64, 1)
NUM_CLASSES = 101

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 20

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


with open('ucfTrainTestlist/classInd.txt', 'r') as f:
    label_map = {l.strip().split(' ')[1]: l.split(' ')[0] for l in f.readlines()}

def video_gen(filepath, video_shape=INPUT_SHAPE):
    with open(filepath, 'r') as f:
        videos_list = [l.strip().split(' ')[0] for l in f.readlines() if l.strip()]
    
    for v in videos_list:
        label = v.split('/')[0]
        label = label_map[label]
        label = int(label) - 1
        
        reader = iio.get_reader(os.path.join('UCF-101', v), format='FFMPEG')
        
        zeros = np.zeros(video_shape)
        for idx, r in enumerate(reader):
            if idx >= video_shape[0]:
                break
            img = cv2.resize(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), 
                    dsize=video_shape[1:-1], interpolation=cv2.INTER_CUBIC)
            zeros[idx] = np.expand_dims(img, -1)
            
        yield zeros,  label

def prepare_dataloader(
    filepath: str,
    loader_type: str = 'train',
    batch_size: int = BATCH_SIZE,
):
    '''Utility function to prepare the dataloader.'''
    dataset = tf.data.Dataset.from_generator(
        partial(video_gen, filepath),
        output_signature=(tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32))
    )
    

    if loader_type == 'train':
        dataset = dataset.shuffle(BATCH_SIZE * 4)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='VALID',
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation='softmax', dtype='float32')(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# TODO: Add TensorBoard support additionally to Neptune
if __name__ == '__main__':
    import sys

    if sys.argv[1] == 'train':
        if '--use-neptune' in sys.argv:
            import neptune.new as neptune
            from neptune.new.integrations.tensorflow_keras import NeptuneCallback
            run = neptune.init(
                project='jegork/thesis',
                api_token=os.getenv('NEPTUNE_API_KEY'),
            )

            
        run['parameters'] = {
            'batch_size': BATCH_SIZE, 
            'input_shape': INPUT_SHAPE, 
            'lr': LEARNING_RATE, 
            'decay': WEIGHT_DECAY, 
            'epochs': EPOCHS, 
            'patch_size': PATCH_SIZE,
            'layer_norm_eps': LAYER_NORM_EPS,
            'projection_dim': PROJECTION_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS
        }

        if '--use-saved-data' in sys.argv:
            trainloader = tf.data.experimental.load('train01.tfrecord')
            testloader = tf.data.experimental.load('test01.tfrecord')
        else:
            trainloader = prepare_dataloader('ucfTrainTestlist/trainlist01.txt', 'train')
            testloader = prepare_dataloader('ucfTrainTestlist/testlist01.txt', 'test')

        neptune_cbk = NeptuneCallback(run=run, base_namespace='training')

        # Initialize model
        model = create_vivit_classifier(
                tubelet_embedder=TubeletEmbedding(
                    embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
                    ),
                positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
                )

        # Compile the model with the optimizer, loss function
        # and the metrics.
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=[
                    keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy'),
                    ],
                )

        # Train the model.
        h = model.fit(trainloader, epochs=EPOCHS, validation_data=testloader, callbacks=[neptune_cbk])

    elif sys.argv[1] == 'prepare':
        trainloader = prepare_dataloader('ucfTrainTestlist/trainlist01.txt', 'train')
        testloader = prepare_dataloader('ucfTrainTestlist/testlist01.txt', 'test')

        tf.data.experimental.save(trainloader, 'train01.tfrecord')
        tf.data.experimental.save(testloader, 'test01.tfrecord')
        
