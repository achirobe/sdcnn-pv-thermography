from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "data"
DETECTION_DIR  = DATA_ROOT / "detection"
DIAGNOSIS_DIR  = DATA_ROOT / "diagnosis"
EXTERNAL_DIR   = PROJECT_ROOT / "external" / "kaggle_irt_pv"
RESULTS_DIR    = PROJECT_ROOT / "results"

IMG_SIZE    = (224, 224)
N_CHANNELS  = 3
BATCH_SIZE  = 32
EPOCHS      = 50
PATIENCE    = 10
LR          = 1e-4
DROPOUT     = 0.5
N_FOLDS     = 5
SEED        = 42

ARCHITECTURES = ["sdcnn", "vgg16", "mobilenetv2", "efficientnet_b0"]
