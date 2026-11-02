"""
Training Script for TrackNet-Cricket Deep Learning Ball Detector

Trains the multi-frame temporal heatmap regression model on synthetic cricket data
and saves checkpoint weights.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cricket_vision.models.ball_detector import TrackNetCricket
from cricket_vision.models.dataset import CricketBallSyntheticDataset


def focal_loss_heatmap(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    Modified Focal Loss for heatmap regression (CornerNet / CenterNet style).
    Penalty-reduced focal loss around ground-truth Gaussian peaks.
    """
    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)

    neg_weights = torch.pow(1.0 - target, beta)

    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def train_ball_detector(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_samples: int = 80,
    save_path: str = "cricket_vision/models/weights/tracknet_cricket.pth"
):
    print("=" * 70)
    print("  TRAINING DEEP LEARNING CRICKET BALL DETECTOR (TrackNetCricket)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Create dataset & dataloader
    print(f"[*] Generating {num_samples} synthetic training delivery triplets...")
    train_dataset = CricketBallSyntheticDataset(num_samples=num_samples, image_size=(640, 360))
    val_dataset = CricketBallSyntheticDataset(num_samples=20, image_size=(640, 360))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TrackNetCricket().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)

            # Combined MSE and Heatmap Loss
            mse_loss = nn.functional.mse_loss(preds, y, reduction="mean") * 100.0
            focal = focal_loss_heatmap(preds, y)
            total_loss = mse_loss + focal

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        scheduler.step()
        train_epoch_loss = running_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                v_preds = model(vx)
                v_mse = nn.functional.mse_loss(v_preds, vy, reduction="mean") * 100.0
                v_focal = focal_loss_heatmap(v_preds, vy)
                val_loss += (v_mse + v_focal).item()

        val_epoch_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path}")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[OK] Training completed in {elapsed:.1f}s. Model weights: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrackNet Cricket Ball Detector")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--samples", type=int, default=60, help="Training dataset samples")
    parser.add_argument("--save-path", type=str, default="cricket_vision/models/weights/tracknet_cricket.pth")
    args = parser.parse_args()

    train_ball_detector(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples,
        save_path=args.save_path
    )
