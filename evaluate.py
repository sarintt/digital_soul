import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import DigitalSoulDataset
from src.model import DigitalSoulModel
import config

# --- SETTINGS ---
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']


def load_resources():
    print("⚙️ Loading Model and Data...")
    device = config.DEVICE

    # 1) Load full dataset for evaluation
    dataset = DigitalSoulDataset()
    print(f"Evaluation dataset size: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2) Init model
    model = DigitalSoulModel().to(device)

    # 3) Load checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "digital_soul_final.pth")
    print(f"📂 Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Handle different save formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print("   ✅ Loaded model from key: 'state_dict'")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("   ✅ Loaded model from key: 'model_state_dict'")
    else:
        model.load_state_dict(checkpoint)
        print("   ✅ Loaded model from raw state_dict")

    model.eval()
    return model, loader, device


def experiment_1_accuracy(model, loader, device):
    print("\n🧪 EXP 1: Accuracy & Scatter Plots...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ling, acou, labels in loader:
            ling, acou = ling.to(device), acou.to(device)

            # Model returns logits -> apply sigmoid to map to [0, 1]
            logits = model(ling, acou)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 1. Overall MAE
    overall_mae = np.mean(np.abs(all_preds - all_labels))
    print(f"   👉 Overall Model MAE: {overall_mae:.4f}")

    # 2. Per-Trait MAE & Scatter Plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    mae_report = {}

    for i, trait in enumerate(TRAITS):
        mae = np.mean(np.abs(all_preds[:, i] - all_labels[:, i]))
        mae_report[trait] = mae

        ax = axes[i]
        # Scatter plot (ground truth vs predicted)
        ax.scatter(all_labels[:, i], all_preds[:, i], alpha=0.3, s=10)
        # Ideal y=x line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)

        ax.set_title(f"{trait}\nMAE: {mae:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ground Truth")
        if i == 0:
            ax.set_ylabel("Predicted")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp1_accuracy_scatter.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"   ✅ Saved scatter plots to {out_path}")

    return all_preds, all_labels, mae_report


def experiment_2_ablation(model, loader, device):
    print("\n🧪 EXP 2: Ablation Study (Sensitivity)...")
    criterion = torch.nn.L1Loss()
    modes = ['Full Fusion', 'Text Only', 'Audio Only']
    report = {}

    with torch.no_grad():
        for mode in modes:
            total_loss = 0.0
            count = 0

            for ling, acou, labels in loader:
                ling, acou, labels = ling.to(device), acou.to(device), labels.to(device)

                # Masking Logic
                if mode == 'Text Only':
                    acou = torch.zeros_like(acou)
                elif mode == 'Audio Only':
                    ling = torch.zeros_like(ling)

                # Model returns logits -> apply sigmoid
                logits = model(ling, acou)
                preds = torch.sigmoid(logits)

                loss = criterion(preds, labels)
                total_loss += loss.item() * len(labels)
                count += len(labels)

            avg_mae = total_loss / count
            report[mode] = avg_mae
            print(f"   👉 {mode} MAE: {avg_mae:.4f}")

    # Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar(report.keys(), report.values())
    plt.title("Impact of Modalities (Lower is Better)")
    plt.ylabel("Mean Absolute Error")

    out_path = os.path.join(RESULTS_DIR, "exp2_ablation_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Saved ablation chart to {out_path}")


def experiment_3_latency(model, device):
    print("\n🧪 EXP 3: Latency & Speed Test...")
    # Dummy Input (Batch 1)
    dummy_ling = torch.randn(1, 768).to(device)

    # 🔥 ใช้ acoustic dim จากโมเดลแทน 13 เดิม
    acou_dim = getattr(model, "acou_input_dim", 40)
    dummy_acou = torch.randn(1, 216, acou_dim).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_ling, dummy_acou)

        times = []
        for _ in range(100):
            start = time.time()
            _ = model(dummy_ling, dummy_acou)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)

    avg_ms = np.mean(times)
    throughput = 1000.0 / avg_ms if avg_ms > 0 else float('inf')

    print(f"   👉 Avg Inference Time: {avg_ms:.2f} ms")
    print(f"   👉 Throughput: {throughput:.0f} req/sec")


def main():
    print("🚀 STARTING FULL SYSTEM EVALUATION")
    print("==================================")

    model, loader, device = load_resources()

    experiment_1_accuracy(model, loader, device)
    experiment_2_ablation(model, loader, device)
    experiment_3_latency(model, device)

    print("\n==================================")
    print(f"🏁 DONE. Check the '{RESULTS_DIR}' folder for your presentation images.")


if __name__ == "__main__":
    main()
