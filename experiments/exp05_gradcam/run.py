"""
Exp 05 — Grad-CAM attention maps.
Loads the best-fold SDCNN detection model and generates a 3×3 overlay grid
showing correctly classified faulty, correctly classified non-faulty,
and misclassified images.
Addresses reviewer objection: black-box CNN, no interpretability.
Requires exp01 to be complete first.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
from src.config import RESULTS_DIR
from src.data.loaders import load_image_paths_and_labels, make_dataset
from src.evaluation.gradcam import generate_gradcam_figure
from src.viz.plots import plot_confusion_matrix

TASK       = "detection"
MODEL_NAME = "sdcnn"

if __name__ == "__main__":
    model_path = RESULTS_DIR / "logs" / f"best_{TASK}_{MODEL_NAME}.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}\nRun exp01 first.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model: {model_path}")

    # Use all images — pick representative examples from the full dataset
    paths, labels = load_image_paths_and_labels(TASK)

    # Run inference on full dataset to get predictions
    ds = make_dataset(paths, labels, augment=False, training=False)
    y_prob = model.predict(ds, verbose=1).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = labels

    print(f"Dataset: {len(paths)} images")
    print(f"Correct: {(y_pred == y_true).sum()}/{len(y_true)}")
    print(f"Misclassified: {(y_pred != y_true).sum()}/{len(y_true)}")

    generate_gradcam_figure(
        model=model,
        val_paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        last_conv_name="conv3",
        save_path=RESULTS_DIR / "figures" / "gradcam_grid.png",
    )

    # Also regenerate confusion matrices from best CV fold for all models
    print("\nRegenerating confusion matrices from best CV fold...")
    for task in ("detection", "diagnosis"):
        for arch in ("sdcnn", "vgg16", "mobilenetv2", "efficientnet_b0"):
            plot_confusion_matrix(task, arch, save=True)

    print("\nExp 05 complete.")
