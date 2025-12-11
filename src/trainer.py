import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.dataset import DigitalSoulDataset
from src.model import DigitalSoulModel
from src.utils import save_checkpoint
import config


def train_model():
    print("🚀 Starting Training Pipeline...")

    # 1. Prepare Data
    full_dataset = DigitalSoulDataset()
    dataset_size = len(full_dataset)
    print(f"📊 Dataset size: {dataset_size} samples")

    # Simple Train/Val split (80/20) with fixed seed for reproducibility
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # 2. Init Model
    model = DigitalSoulModel().to(config.DEVICE)

    # Optimizer: AdamW + weight decay แรงขึ้นนิดหน่อย
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-2,  
    )

    # Scheduler: ลด lr เมื่อ val_loss ไม่ดีขึ้น
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Loss ใช้เทรนจริง ๆ → SmoothL1 (Huber)
    criterion_train = nn.SmoothL1Loss(beta=0.1)

    # Metric สำหรับรีพอร์ต MAE (L1Loss เฉย ๆ)
    mae_metric = nn.L1Loss()

    # --- TRACKING VARIABLES ---
    best_val_mae = float('inf')
    patience_counter = 0
    PATIENCE_LIMIT = 15  # early stopping ถ้าไม่ดีขึ้นติดต่อกัน

    # 3. Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_mae = 0.0

        for ling, acou, labels in train_loader:
            ling = ling.to(config.DEVICE)
            acou = acou.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            # Modality Dropout (train-time only)
            if getattr(config, "MODALITY_DROPOUT", False):
                r_text = torch.rand(1).item()
                r_audio = torch.rand(1).item()

                # ปิด text ทั้ง batch บางครั้ง
                if r_text < getattr(config, "DROP_TEXT_PROB", 0.0):
                    ling = torch.zeros_like(ling)

                # ปิด audio ทั้ง batch บางครั้ง
                if r_audio < getattr(config, "DROP_AUDIO_PROB", 0.0):
                    acou = torch.zeros_like(acou)

            # Model returns logits (raw scores)
            logits = model(ling, acou)

            # Map logits -> [0,1]
            outputs = torch.sigmoid(logits)

            # Training loss (SmoothL1)
            loss = criterion_train(outputs, labels)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # สะสม loss + MAE
            total_train_loss += loss.item()
            total_train_mae += mae_metric(outputs, labels).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mae = total_train_mae / len(train_loader)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0.0
        total_val_mae = 0.0

        with torch.no_grad():
            for ling, acou, labels in val_loader:
                ling = ling.to(config.DEVICE)
                acou = acou.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # validation phase: ห้ามทำ modality dropout
                logits = model(ling, acou)
                outputs = torch.sigmoid(logits)

                # ใช้ SmoothL1 เป็น val_loss ด้วย เพื่อ match training
                total_val_loss += criterion_train(outputs, labels).item()
                total_val_mae += mae_metric(outputs, labels).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mae = total_val_mae / len(val_loader)

        # Update Scheduler ด้วย val_loss
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] - "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}"
        )

        # --- SAVE BEST MODEL (Val MAE) ---
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            print(f"   🔥 New Best Model! (Val MAE: {best_val_mae:.4f}) -> Saving...")
            save_checkpoint(model, optimizer, filename="digital_soul_final.pth")
        else:
            patience_counter += 1
            print(f"   😴 No improvement. Patience: {patience_counter}/{PATIENCE_LIMIT}")

        # --- EARLY STOPPING ---
        if patience_counter >= PATIENCE_LIMIT:
            print(f"\n🛑 Early Stopping triggered! No improvement for {PATIENCE_LIMIT} epochs.")
            print(f"   Best Validation MAE was: {best_val_mae:.4f}")
            break

    print("🏁 Training Complete!")


if __name__ == "__main__":
    train_model()
