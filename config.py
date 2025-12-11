import os
import torch

# --- PATHS ---
# Automatic root directory detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
AUDIO_DIR = os.path.join(DATA_DIR, "processed_audio")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

LABELS_PATH = os.path.join(DATA_DIR, "ground_truth", "train-annotation", "annotation_training.pkl")

# --- DATA PROCESSING ---
SAMPLE_Rate = 16000
MAX_DURATION = 15
DATA_LIMIT = None 

# --- MODEL HYPERPARAMETERS ---
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_TEXT_LEN = 2048

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 500
HIDDEN_DIM = 64 # Size of the acoustic LSTM memory
DROPOUT = 0.2

# --- HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Helper to make directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODALITY_DROPOUT = True      # เปิด/ปิด ฟีเจอร์นี้ได้
DROP_TEXT_PROB = 0.15        # โอกาสปิด text ต่อ batch (แนะนำ 0.1–0.2)
DROP_AUDIO_PROB = 0.15       # โอกาสปิด audio ต่อ batch