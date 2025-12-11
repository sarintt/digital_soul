# Digital Soul 🧠🔊

**A Multi-Modal AI Approach to Personality Prediction from Speech**

Digital Soul is a deep learning system that predicts a speaker’s **Big Five personality traits**  
(Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from short voice clips.

The model fuses:

- **Linguistic analysis** (what you say – ModernBERT)
- **Acoustic analysis** (how you say it – MFCC-based prosody features)

to build a holistic *"personality fingerprint"*.

## 🌟 Key Features

### ✔ Modern Multi-Modal Architecture
- **Text Encoder:** ModernBERT (CLS embedding, 768-dim)
- **Audio Encoder:** BiLSTM + Attention using **40-dim enriched MFCC features**:
  - MFCC (13)
  - Δ MFCC (13)
  - Δ² MFCC (13)
  - RMS energy (1)

### ✔ Advanced Fusion Technique
- **Gated Fusion Layer** learns how to balance linguistic vs acoustic information dynamically.  
  → Fusion performs **better than text-only for the first time**.

### ✔ Enhanced Training Stability
- AdamW optimizer  
- SmoothL1Loss  
- Gradient clipping  
- ReduceLROnPlateau scheduler  
- Early stopping  
- **Modality dropout** (forces model to learn both modalities)

### ✔ Full Scientific Evaluation
- MAE computation  
- Modality ablation (Text-only, Audio-only, Fusion)
- Scatter plots  
- Latency benchmarking

### ✔ Interactive Demo
- Upload a voice clip
- Real-time transcription + feature extraction
- Visual personality radar chart via Gradio

## 🛠️ Installation

This project uses `uv` for ultra-fast dependency management.

### Prerequisites

* **Python 3.11** (Required for Librosa/Numba compatibility)
* **FFmpeg** installed on your system (Required for Whisper audio processing).
  * *Windows:* `winget install -e --id Gyan.FFmpeg`
  * *Mac:* `brew install ffmpeg`
  * *Linux:* `sudo apt install ffmpeg`

### Setup Steps

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/sarinntt/dsi442_2025.git](https://github.com/sarinntt/dsi442_2025.git)
   cd dsi442_2025
   ```

2. **Initialize environment with uv:**

    ```bash
    uv sync --no-install-project
    ```

*This installs PyTorch (GPU enabled), Transformers, Librosa, and all other dependencies.*

## 🚀 Usage Pipeline
The entire workflow is managed via the `main.py` CLI.

1. **Data Preparation** (`prep`)
Converts raw video files (`.mp4`) into standardized audio files (`.wav`, 16kHz, Mono).
* *Input:* `data/raw_videos/`
* *Output:* `data/processed_audio/`

    ```bash
    uv run main.py prep
    ```

2. **Feature Extraction** (`extract`)
Extracts mathematical features from the audio.
* **Linguistic**: Transcribes audio (Whisper) -> Tokenizes -> Embeds via ModernBERT (768 dim).
* **Acoustic**: Using Librosa per frame:

| Feature Type | Dim |
|--------------|-----|
| MFCC | 13 |
| Δ-MFCC | 13 |
| Δ²-MFCC | 13 |
| RMS Energy | 1 |
| **Total** | **40** |


* *Output:* `.npy` files in `data/features/`

1. **Model Training** (`train`)
Trains the Multi-Modal Neural Network using the extracted features.
* **Config:** AdamW optimizer, SmoothL1Loss, ReduceLROnPlateau scheduler, gradient clipping, modality dropout, early stopping.
* *Output:* Saves the best model to `checkpoints/digital_soul_final.pth.`

    ```bash
    uv run main.py train
    ```

1. **Evaluation** (`evaluate`)
Runs the complete scientific evaluation pipeline, including overall MAE, per-trait MAE, modality ablation (Text-Only, Audio-Only, Full Fusion), scatter plots, and latency benchmarking. All evaluation charts are saved in `results/`.

    ```bash
    uv run evaluate.py
    ```

## 🎮 Interactive Demo
Launch the web interface to test the model with your own voice.

    ```bash
    uv run app.py
    ```

* Open the local URL (e.g., `http://127.0.0.1:7860`) in your browser.
* Upload a `.wav` or `.mp3` file.
* View your Personality Radar Chart.

## 📂 Project Structure
```
Digital_Soul/
├── app.py                 # Gradio Web Application
├── config.py              # Hyperparameters & Settings
├── main.py                # Command-line Pipeline Controller
├── evaluate.py            # Model Evaluation & Ablation
├── checkpoints/           # Saved Model Weights
├── results/               # Charts & Plots
│
└── src/
    ├── dataset.py         # PyTorch Dataset Loader
    ├── features.py        # Whisper + ModernBERT + MFCC40 Extractor
    ├── model.py           # BiLSTM + Attention + Gated Fusion Model
    ├── preprocessing.py   # Audio Conversion Utilities
    ├── trainer.py         # Training Loop (AdamW + SmoothL1Loss)
    └── utils.py           # Helper Functions

## 📜 Dataset
This project uses the **ChaLearn First Impressions V2** dataset.
* **Size:** ~10,000 video clips (15s average).
* **Labels:** Big Five Personality Traits (0.0 - 1.0).
*Note: Dataset must be obtained via official challenge channels.*

## 📄 License
This project is for academic purposes.