# src/config.py


import torch
import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
IMAGE_DIR       = os.path.join(DATA_DIR, "images")
CSV_PATH        = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
KB_PATH         = os.path.join(BASE_DIR, "knowledge_base", "thoracic_kb.txt")

# Creating directories if they don't exist
for d in [CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# The 14 Disease Labels

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

NUM_CLASSES     = len(LABELS)   # 14

# Image Settings 
IMAGE_SIZE      = 224
CHANNELS        = 3

# Training Settings 
BATCH_SIZE      = 32
NUM_EPOCHS      = 10
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5

# Split Ratios 
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.10
TEST_RATIO      = 0.20

# Decision Threshold 
THRESHOLD       = 0.5

# Device 
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Using CPU")

# Random Seed
SEED = 42