from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

EDA_RESULTS_DIR = RESULTS_DIR / "eda"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"

ZIP_CANDIDATE_NAMES = (
    "Archive(2).zip",
    "Archive (2).zip",
    "archive (2).zip",
    "archive(2).zip",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# RAF-DB basic emotion labels. The EDA still detects classes automatically;
# this map only adds human-readable names when the canonical labels are found.
RAFDB_LABEL_MAP = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

RANDOM_SEED = 42

NUM_CLASSES = 7
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
VALIDATION_SIZE = 0.15

TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
TUNING_RESULTS_DIR = RESULTS_DIR / "tuning"
ARTICLE_RESULTS_DIR = RESULTS_DIR / "article"

TRAINING_FIGURES_DIR = FIGURES_DIR / "training"
EVALUATION_FIGURES_DIR = FIGURES_DIR / "evaluation"
EXPLAINABILITY_FIGURES_DIR = FIGURES_DIR / "explainability"

SUPPORTED_ARCHITECTURES = (
    "custom_cnn",
    "vgg16",
    "resnet50",
    "mobilenetv2",
    "efficientnetb0",
    "densenet121",
    "qcs",
    "poster_v2",
    "swin_face",
    "deit",
)

SUPPORTED_SCENARIOS = ("scratch", "transfer", "fine_tuning")
