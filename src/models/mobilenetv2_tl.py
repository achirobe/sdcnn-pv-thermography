import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from src.config import IMG_SIZE, N_CHANNELS, DROPOUT, LR


def build_mobilenetv2(num_classes: int = 1) -> Model:
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=(*IMG_SIZE, N_CHANNELS))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="mobilenetv2_tl")
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
