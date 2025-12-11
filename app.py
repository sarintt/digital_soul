import os
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import gradio as gr

from src.features import FeatureEngine
from src.model import DigitalSoulModel
import config


# ===========================
#  SETUP & MODEL LOADING
# ===========================
device = config.DEVICE
print(f"🚀 Launching Digital Soul on {device}...")

model = DigitalSoulModel().to(device)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "digital_soul_final.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # รองรับหลายรูปแบบการ save
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("✅ Loaded weights from key: 'state_dict'")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("✅ Loaded weights from key: 'model_state_dict'")
        else:
            state_dict = checkpoint
            print("✅ Loaded weights directly from checkpoint dict")
    else:
        state_dict = checkpoint
        print("✅ Loaded raw state_dict")

    model.load_state_dict(state_dict)
    print("✅ Model weights loaded!")
else:
    print("⚠️ No trained model found — using random weights.")

model.eval()
engine = FeatureEngine()

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


# ===========================
#  RADAR CHART
# ===========================
def generate_radar_chart(scores: np.ndarray):
    """
    Radar chart with clean white+purple theme.

    Expect scores in range [0, 1].
    Inside this function we scale to 0–100 for plotting.
    """

    # Ensure chart uses same font family
    plt.rcParams["font.family"] = "Inter"

    labels = [
        "Openness",
        "Extraversion",
        "Neuroticism",
        "Conscientiousness",
        "Agreeableness",
    ]

    # Model order [O, C, E, A, N] → [O, E, N, C, A]
    o, c, e, a, n = scores
    scores_ordered = np.array([o, e, n, c, a]) * 100.0  # scale 0–1 -> 0–100
    values = np.concatenate((scores_ordered, [scores_ordered[0]]))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))

    # Background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Orientation
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Radial grid
    grid_vals = [25, 50, 75, 100]
    ax.set_rgrids(grid_vals, angle=90, fontsize=10, color="#9ca3af")
    ax.set_ylim(0, 100)

    # Grid rings
    grid_color = "#e5e7eb"
    for r in grid_vals + [0]:
        ax.plot(angles, [r] * len(angles), color=grid_color, linewidth=1, alpha=0.85)

    # Radar polygon
    ax.fill(angles, values, color="#c4b5fd", alpha=0.45)
    ax.plot(angles, values, color="#7c3aed", linewidth=2)

    # Axis labels – clearer, not overlapping lines
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight=500, color="#3f3f46", ha="center")
    ax.tick_params(axis="x", pad=12)

    ax.spines["polar"].set_color(grid_color)

    fig.tight_layout()
    return fig


# ===========================
#  RESULT CARD HTML
# ===========================
def build_html_summary(text: str, scores: np.ndarray) -> str:
    """
    scores: expected in [0,1] for each trait.
    We convert to 0–100 for display.
    """
    rows = ""
    scores_percent = scores * 100.0  # scale to percentage

    for trait, score in zip(TRAITS, scores_percent):
        rows += f"""
        <div class="score-row">
            <span>{trait}</span>
            <span>{score:.2f}%</span>
        </div>
        """

    return f"""
    <div class="card result-card">
        <div class="section-title" style="font-weight:700;">Script</div>
        <p class="Script-text">"{text}"</p>

        <div class="section-title" style="margin-top:18px; font-weight:700;">Personality Scores</div>
        {rows}
        <p class="disclaimer">
            These values are model-based estimates and should be interpreted
            as soft tendencies, not fixed labels.
        </p>
    </div>
    """


# ===========================
#  PREDICTION PIPELINE
# ===========================
def predict_personality(audio_file):
    if audio_file is None:
        return None, """
        <div class='card result-card empty-card'>
            Upload a short voice clip and click <b>Analyze Personality</b> to see your Digital Soul.
        </div>
        """

    # 1) Acoustic features: MFCC13 + Δ + Δ² + RMS = 40 dims (ให้ตรงกับตอนเทรน)
    y, sr = librosa.load(audio_file, sr=config.SAMPLE_Rate, duration=config.MAX_DURATION)
    y = librosa.util.fix_length(y, size=config.SAMPLE_Rate * config.MAX_DURATION)

    # MFCC พื้นฐาน 13 ตัว → (13, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Delta / Delta^2 → (13, T)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # RMS energy → (1, T)
    rms = librosa.feature.rms(y=y)

    # รวมเป็น (40, T)
    feats = np.concatenate([mfcc, delta1, delta2, rms], axis=0)  # (40, T)

    # แปลงเป็น tensor shape (B, T, 40) ให้ตรงกับ DigitalSoulModel (LayerNorm(40))
    acou_t = (
        torch.tensor(feats, dtype=torch.float32)  # (40, T)
        .transpose(0, 1)                          # (T, 40)
        .unsqueeze(0)                             # (1, T, 40)
        .to(device)
    )

    # 2) Linguistic features (Whisper + ModernBERT)
    res = engine.whisper.transcribe(audio_file)
    text = res.get("text", "").strip() or "silence"

    inputs = engine.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_TEXT_LEN,
    ).to(device)

    with torch.no_grad():
        text_outputs = engine.text_model(**inputs)
        # CLS token representation
        ling_vec = (
            text_outputs.last_hidden_state[:, 0, :]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

    # 3) Prepare tensors for model
    ling_t = torch.tensor(ling_vec, dtype=torch.float32).unsqueeze(0).to(device)

    # 4) Predict scores
    with torch.no_grad():
        # model returns logits
        logits = model(ling_t, acou_t)

        # แปลง logits → [0,1]
        probs = torch.sigmoid(logits)

        # แปลงเป็น numpy array (shape: (5,))
        scores = probs.detach().cpu().numpy()[0]

    # คืน radar chart + HTML summary
    return generate_radar_chart(scores), build_html_summary(text, scores)


#  THEME & CSS
purple_theme = gr.themes.Soft(primary_hue="purple", neutral_hue="gray")

custom_css = """
body { background:#f4f4ff; }
body * { font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif !important; }

/* Header */
.app-header { text-align:center; margin:6px 0 18px 0; }
.app-title { font-size:32px; font-weight:700; color:#111827; }
.app-subtitle { font-size:14px; color:#6b7280; }

/* Card */
.card {
    background:#ffffff;
    border-radius:18px;
    padding:18px 20px;
    box-shadow:0 10px 26px rgba(15,23,42,0.06);
    border:1px solid #e5e7eb;
}

/* Info card title */
.info-card-title { font-size:16px; font-weight:600; margin-bottom:10px; }

/* Result card */
.result-card { min-height:260px; }
.empty-card {
    display:flex;
    align-items:center;
    justify-content:center;
    text-align:center;
    color:#9ca3af;
}

/* Section text */
.section-title { font-size:16px; color:#111827; margin-bottom:6px; }
.transcript-text { font-size:13px; color:#4b5563; line-height:1.4; }

/* Table rows */
.score-row {
    display:flex;
    justify-content:space-between;
    font-size:13px;
    font-weight:500;
    padding:5px 0;
    border-bottom:1px solid #e5e7eb;
}
.score-row:last-child { border-bottom:none; }
.disclaimer { margin-top:10px; font-size:11px; color:#9ca3af; }

/* Analyze button: pill style */
#analyze-btn {
    background:linear-gradient(90deg,#7c3aed,#a855f7);
    color:#ffffff;
    border-radius:999px;
    padding:12px 0;
    font-weight:600;
    border:none;
}

/* Center radar plot – no card background */
#radar-center {
    display:flex;
    justify-content:center;
    align-items:center;
}

/* Make the gr.Plot container transparent */
#radar-center > div {
    background:transparent !important;
    box-shadow:none !important;
    border:none !important;
}
"""


# ===========================
#  UI LAYOUT
# ===========================
with gr.Blocks(theme=purple_theme, css=custom_css) as iface:

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">Digital Soul</div>
        <div class="app-subtitle">Voice-based Big Five personality dashboard</div>
    </div>
    """)

    with gr.Row(equal_height=True):

        # LEFT: Big Five explanation
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="card">
                <div class="info-card-title">Big Five Overview</div>
                <p style="font-size:13px; color:#4b5563;">
                    Digital Soul estimates your personality along five continuous traits:
                </p>
                <ul style="list-style:none; padding-left:0; font-size:13px; color:#4b5563; line-height:1.5;">
                    <li><b>Openness</b> – curiosity & creativity</li>
                    <li><b>Conscientiousness</b> – disciplined & reliable</li>
                    <li><b>Extraversion</b> – social & energetic</li>
                    <li><b>Agreeableness</b> – cooperative & empathetic</li>
                    <li><b>Neuroticism</b> – emotional sensitivity</li>
                </ul>
            </div>
            """)

        # MIDDLE: centered chart
        with gr.Column(scale=2):
            gr.HTML("""
            <div style="text-align:center; margin-bottom:10px; font-size:18px; font-weight:600;">
                Your Results
            </div>
            """)
            radar_output = gr.Plot(elem_id="radar-center")

        # RIGHT: upload + analysis
        with gr.Column(scale=1.3):
            audio_input = gr.Audio(
                type="filepath",
                label="Upload voice clip",
                show_label=True,
            )
            analyze_btn = gr.Button("Analyze Personality", elem_id="analyze-btn")
            result_html = gr.HTML(
                "<div class='card result-card empty-card'>Upload a voice clip to begin.</div>"
            )

    analyze_btn.click(
        fn=predict_personality,
        inputs=audio_input,
        outputs=[radar_output, result_html],
    )

if __name__ == "__main__":
    iface.launch()
