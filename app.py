import gradio as gr
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
from src.features import FeatureEngine
from src.model import DigitalSoulModel
import config

# --- SETUP ---
device = config.DEVICE
print(f"🚀 Launching App on {device}...")

# Load Model
model = DigitalSoulModel().to(device)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "digital_soul_final.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model weights loaded!")
else:
    print("⚠️ No trained model found! Using random weights.")

model.eval()

# Initialize Feature Engine (Loads ModernBERT now)
engine = FeatureEngine()

# --- HELPER FUNCTIONS ---
def generate_radar_chart(scores):
    labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    scores = np.concatenate((scores, [scores[0]]))
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='#00ffcc', alpha=0.25)
    ax.plot(angles, scores, color='#00cc99', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    plt.title("Digital Soul Fingerprint", size=15, color='#333333', y=1.1)
    return fig

def predict_personality(audio_file):
    if audio_file is None:
        return None, "Please upload an audio file."

    try:
        # 1. Acoustic Features
        y, sr = librosa.load(audio_file, sr=config.SAMPLE_Rate, duration=config.MAX_DURATION)
        target_len = config.SAMPLE_Rate * config.MAX_DURATION
        if len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
        else:
            y = y[:target_len]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 2. Linguistic Features (ModernBERT)
        res = engine.whisper.transcribe(audio_file)
        text = res['text'].strip() or "silence"
        
        # Use new config context length
        inputs = engine.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=config.MAX_TEXT_LEN
        ).to(device)
        
        with torch.no_grad():
            outputs = engine.text_model(**inputs)
        
        # Extract [CLS] token (index 0)
        ling_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

        # 3. Prepare Tensors
        acou_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
        acou_t = acou_t.transpose(1, 2)
        
        ling_t = torch.tensor(ling_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # 4. Predict
        with torch.no_grad():
            prediction = model(ling_t, acou_t).cpu().numpy()[0]

        chart = generate_radar_chart(prediction)
        
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        summary = f"📝 Transcript (ModernBERT analyzed {len(text.split())} words):\n\"{text}\"\n\n📊 Personality Profile:\n"
        for t, s in zip(traits, prediction):
            summary += f"- {t}: {s:.2f}\n"

        return chart, summary

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- UI SETUP ---
iface = gr.Interface(
    fn=predict_personality,
    inputs=gr.Audio(type="filepath", label="Upload Voice"),
    outputs=[gr.Plot(label="Personality Radar"), gr.Textbox(label="Analysis")],
    title="Digital Soul",
    description="Upload an audio clip. We use ModernBERT to analyze the transcript and Librosa for vocal cues."
)

if __name__ == "__main__":
    iface.launch()