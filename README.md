# Digital Soul 🧠🔊

**A Multi-Modal AI Approach to Personality Prediction from Speech**

"Digital Soul" is a deep learning project that predicts a person's **Big Five personality traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from short audio clips. By fusing linguistic analysis (what is said) with acoustic analysis (how it is said), the model creates a holistic "personality fingerprint."

## 🌟 Key Features

* **Multi-Modal Architecture:** Combines text embeddings (ModernBERT) and vocal features (MFCCs via Librosa).
* **State-of-the-Art Models:** Powered by **OpenAI Whisper** for transcription and **ModernBERT** (8k context) for linguistic analysis.
* **Acoustic Intelligence:** Uses an **LSTM** network to capture prosody, pitch, and energy dynamics.
* **Interactive Demo:** A Gradio-based web app that generates a real-time "Digital Soul" radar chart.

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
   git clone [https://github.com/SatNichapon/dsi442_2025.git](https://github.com/SatNichapon/dsi442_2025.git)
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
* **Acoustic**: Extracts MFCCs via Librosa (13 dim).
* *Output:* `.npy` files in `data/features/`

3. **Model Training** (`train`)
Trains the Multi-Modal Neural Network using the extracted features.
* **Config:** 50 Epochs, Adam Optimizer, Early Stopping.
* *Output:* Saves the best model to `checkpoints/digital_soul_final.pth.`

    ```bash
    uv run main.py train
    ```

4. **Evaluation** (`evaluate`)
Runs a full suite of scientific tests (Accuracy, Ablation, Latency, Correlation) and generates charts in `results/`.

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