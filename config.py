from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# данные
DATA_DIR = PROJECT_DIR / "cleaned"
METADATA_PATH = DATA_DIR / "audio_metadata_cleaned.csv"
FT_DATA_DIR = PROJECT_DIR / "cleaned_subset_200"
FT_METADATA_PATH = FT_DATA_DIR / "audio_metadata_cleaned.csv"

# выход
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "vit_lfpe_ssl_best.pt"
FT_BEST_CKPT = CHECKPOINT_DIR / "vit_lfpe_bat_finetune_best.pt"

# эксперимент
RANDOM_SEED = 42
VAL_TEST_SIZE = 0.15
BATCH_SIZE = 16
NUM_WORKERS = 0

# SSL pretrain
MASK_RATIO = 0.75
SEM_UTTERANCE_WEIGHT = 1.0
SEM_CONTRASTIVE_WEIGHT = 0.25
CONTRASTIVE_VIEW_FRAC = 0.65  # overlap = 2 * frac - 1

BASE_LR = 1e-4
MAX_EPOCHS = 50
PATIENCE = 12
WARMUP_EPOCHS = 5

# fine-tune
LOAD_SSL_PRETRAIN = True
FT_MAX_EPOCHS = 40
FT_PATIENCE = 12
FT_ENCODER_LR_SSL = 1e-4
FT_ENCODER_LR_NO_SSL = 5e-4
FT_HEAD_LR = 3e-4
FT_MIXUP_ALPHA = 0.2
