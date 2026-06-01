import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

from src.config import IMG_SIZE, N_CHANNELS, DROPOUT, LR


def build_efficientnet_b0(num_classes: int = 1) -> Model:
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(*IMG_SIZE, N_CHANNELS))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    # EfficientNet expects [0, 255]; the base includes internal rescaling
    x = base(inputs * 255.0, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="efficientnet_b0_tl")
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
