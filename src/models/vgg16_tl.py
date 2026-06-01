import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

from src.config import IMG_SIZE, N_CHANNELS, DROPOUT, LR


def build_vgg16(num_classes: int = 1) -> Model:
    base = VGG16(include_top=False, weights="imagenet",
                 input_shape=(*IMG_SIZE, N_CHANNELS))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = tf.keras.applications.vgg16.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="vgg16_tl")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
        ],
    )
    return model
