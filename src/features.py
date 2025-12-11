import os
import numpy as np
import librosa
import torch
import whisper
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import config

class FeatureEngine:
    def __init__(self):
        self.device = config.DEVICE
        print(f"Initializing Feature Engine on {self.device}...")
        
        # Load Models
        # Whisper
        self.whisper = whisper.load_model("base", device=self.device)
        
        # ModernBERT / RoBERTa (ตาม config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.text_model = AutoModel.from_pretrained(config.MODEL_NAME).to(self.device)
        self.text_model.eval()

    def process_file(self, audio_path, filename):
       
        ling_path = os.path.join(config.FEATURES_DIR, filename.replace(".wav", "_ling.npy"))
        acou_path = os.path.join(config.FEATURES_DIR, filename.replace(".wav", "_acou.npy"))

        # ถ้ามีทั้ง ling + acou แล้ว ข้าม
        if os.path.exists(ling_path) and os.path.exists(acou_path):
            return  # Skip

        # --- ACOUSTIC (Librosa) ---
        # Load max MAX_DURATION sec
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_Rate, duration=config.MAX_DURATION)

        # Pad / trim ให้ยาวเท่ากันเสมอ (สำคัญมากสำหรับ LSTM)
        target_len = config.SAMPLE_Rate * config.MAX_DURATION
        if len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
        else:
            y = y[:target_len]
        
        # 1) MFCC พื้นฐาน
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)           # (13, T)

        # 2) Delta MFCC (ความเปลี่ยนแปลงเวลา)
        mfcc_delta = librosa.feature.delta(mfcc)                     # (13, T)

        # 3) Delta-Delta MFCC (ความเปลี่ยนแปลงของ delta)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)           # (13, T)

        # 4) RMS Energy (ความดังต่อเฟรม)
        rms = librosa.feature.rms(y=y)                               # (1, T)

        # Stack ให้ได้ shape (40, T)
        acou_feat = np.vstack([mfcc, mfcc_delta, mfcc_delta2, rms])  # (13+13+13+1 = 40, T)

        # Save acoustic feature
        np.save(acou_path, acou_feat)

        # --- LINGUISTIC (Whisper + RoBERTa/ModernBERT) ---
        # 1. Transcribe ด้วย Whisper
        res = self.whisper.transcribe(audio_path)
        text = res.get("text", "").strip() or "silence"
        
        # 2. Embed ด้วย Transformer (CLS Token)
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=config.MAX_TEXT_LEN
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)

        last_hidden_state = outputs.last_hidden_state   # (B, L, H)
        vec = last_hidden_state[:, 0, :].cpu().numpy().squeeze()  # CLS -> (H,)

        np.save(ling_path, vec)

    def run_batch(self):
        print("Starting Feature Extraction...")
        audio_files = [f for f in os.listdir(config.AUDIO_DIR) if f.endswith(".wav")]
        
        # Respect limit if needed
        if config.DATA_LIMIT:
            audio_files = audio_files[:config.DATA_LIMIT]

        for f in tqdm(audio_files):
            path = os.path.join(config.AUDIO_DIR, f)
            try:
                self.process_file(path, f)
            except Exception as e:
                print(f"Error extracting {f}: {e}")
