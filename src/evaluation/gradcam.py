"""
Gradient-weighted Class Activation Mapping (Grad-CAM).
Selvaraju et al., 2017 — implemented via tf.GradientTape.
Applied to the SDCNN model's final convolutional layer (conv3).
"""
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.config import IMG_SIZE, RESULTS_DIR


def _load_image(path: str) -> np.ndarray:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def compute_gradcam(model, img_array: np.ndarray,
                    last_conv_name: str = "conv3",
                    class_idx: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (heatmap_224x224, overlay_224x224x3).
    class_idx=1 targets the positive/faulty class.
    """
    feat_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )

    img_tensor = tf.cast(img_array[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = feat_model(img_tensor, training=False)
        tape.watch(conv_out)
        score = preds[:, class_idx] if preds.shape[-1] > 1 else preds[:, 0]

    grads = tape.gradient(score, conv_out)           # (1, H, W, C)
    pooled = tf.reduce_mean(grads, axis=(1, 2))[0]  # (C,)

    cam = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    cam = tf.nn.relu(cam).numpy()

    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    heatmap = cv2.resize(cam, IMG_SIZE)

    # Jet colourmap overlay
    jet = plt.cm.jet(heatmap)[..., :3]              # (H, W, 3) float
    overlay = np.clip(0.55 * img_array + 0.45 * jet, 0, 1)

    return heatmap, overlay


def _pick_samples(val_paths, y_true, y_pred, label, correct, n=3):
    """Pick n image paths matching given label and correct/incorrect prediction."""
    mask = (y_true == label) & ((y_pred == y_true) == correct)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []
    chosen = idxs[:n]
    return [val_paths[i] for i in chosen]


def generate_gradcam_figure(model, val_paths, y_true, y_pred,
                             last_conv_name: str = "conv3",
                             save_path: Path = None):
    """
    Build a 3-panel × 3-column figure:
      Row 1: 3 correctly classified faulty (label=1)
      Row 2: 3 correctly classified non-faulty (label=0)
      Row 3: 3 misclassified images (any label, wrong prediction)
    Each column: original | heatmap | overlay
    """
    correct_faulty    = _pick_samples(val_paths, y_true, y_pred, label=1, correct=True)
    correct_nonfaulty = _pick_samples(val_paths, y_true, y_pred, label=0, correct=True)
    misclassified     = _pick_samples(val_paths, y_true, y_pred, label=1, correct=False)
    # pad if not enough misclassified
    if len(misclassified) < 3:
        misclassified += _pick_samples(val_paths, y_true, y_pred, label=0, correct=False)
    misclassified = misclassified[:3]

    groups = [
        ("Correct — Faulty",     correct_faulty[:3]),
        ("Correct — Non-faulty", correct_nonfaulty[:3]),
        ("Misclassified",        misclassified[:3]),
    ]

    n_rows = len(groups)
    n_cols = 9   # 3 images × 3 panels each
    fig = plt.figure(figsize=(18, n_rows * 4))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.3)

    col_titles = ["Original", "Heatmap", "Overlay"]

    for row_idx, (row_label, paths_group) in enumerate(groups):
        for img_idx, path in enumerate(paths_group):
            img = _load_image(path)
            heatmap, overlay = compute_gradcam(model, img, last_conv_name, class_idx=1)

            panels = [img, heatmap, overlay]
            for panel_idx, panel in enumerate(panels):
                col = img_idx * 3 + panel_idx
                ax = fig.add_subplot(gs[row_idx, col])
                if len(panel.shape) == 2:
                    ax.imshow(panel, cmap="jet")
                else:
                    ax.imshow(panel)
                ax.axis("off")
                if row_idx == 0 and img_idx == 0:
                    ax.set_title(col_titles[panel_idx], fontsize=9, pad=4)

        # row label on first image
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.set_ylabel(row_label, fontsize=9, labelpad=6)

    plt.suptitle("Grad-CAM Attention Maps — SDCNN Detection Model",
                 fontsize=12, y=1.01)

    if save_path is None:
        save_path = RESULTS_DIR / "figures" / "gradcam_grid.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[gradcam] Saved → {save_path}")
