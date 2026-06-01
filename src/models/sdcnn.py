import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import IMG_SIZE, N_CHANNELS, DROPOUT, LR


def build_sdcnn(num_classes: int = 1) -> Model:
    inputs = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(name="pool1")(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(name="pool2")(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(name="pool3")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="sdcnn")
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
