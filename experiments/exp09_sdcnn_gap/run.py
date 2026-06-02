"""
Exp 09 — SDCNN with GlobalAveragePooling2D head instead of Flatten.

Reviewer response: the Flatten->Dense(128) head holds ~11M of SDCNN's 11.17M
parameters, making the "lightweight" claim weaker than the modern baselines.
Replacing Flatten with GAP drops the model to ~0.11M parameters. This script
re-runs the identical 5-fold CV protocol so the comparison is apples-to-apples.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import IMG_SIZE, N_CHANNELS, DROPOUT, LR
from src.training.cv_runner import run_cv


def build_sdcnn_gap(num_classes: int = 1) -> Model:
    inputs = tf.keras.Input(shape=(*IMG_SIZE, N_CHANNELS))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(name="pool1")(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(name="pool2")(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(name="pool3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="sdcnn_gap")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model


if __name__ == "__main__":
    # Report parameter count for the manuscript
    m = build_sdcnn_gap()
    total = m.count_params()
    print(f"\nSDCNN-GAP total parameters: {total:,}\n", flush=True)

    for task in ("detection", "diagnosis"):
        run_cv(task=task, model_name="sdcnn_gap", build_fn=build_sdcnn_gap)
    print("\nExp 09 complete.", flush=True)
