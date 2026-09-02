import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataloaders
from model import DentalViT, DualLoss


@torch.no_grad()
def compute_seg_metrics(logits, mask, threshold=0.5):
    """Pixel-level metrics; never shrink caries masks to the 14x14 token grid."""
    pred = torch.sigmoid(logits) >= threshold
    target = mask.bool()
    tp = (pred & target).sum(dtype=torch.float32)
    fp = (pred & ~target).sum(dtype=torch.float32)
    fn = (~pred & target).sum(dtype=torch.float32)
    eps = torch.finfo(torch.float32).eps
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (pred == target).float().mean()
    return dice.item(), iou.item(), accuracy.item()


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_frac=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_frac = min_lr_frac
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr_frac + (1 - self.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")
        self.model = DentalViT(pretrained=not cfg.no_pretrained, dropout=cfg.dropout).to(self.device)
        self.criterion = DualLoss(
            alpha=cfg.alpha, beta=cfg.beta, pos_weight=cfg.pos_weight
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = WarmupCosineSchedule(self.optimizer, cfg.warmup_epochs, cfg.epochs)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            cfg.data_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.best_val_dice = -1.0

    def _run_epoch(self, loader, train=False, epoch=0):
        self.model.train(train)
        totals = {"loss": 0.0, "bce": 0.0, "mse": 0.0, "dice": 0.0, "iou": 0.0, "acc": 0.0}
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train):
                logits, adjacency = self.model(images)
                loss, bce, mse = self.criterion(logits, adjacency, masks)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            dice, iou, acc = compute_seg_metrics(logits, masks)
            for key, value in (("loss", loss.item()), ("bce", bce.item()), ("mse", mse.item()),
                               ("dice", dice), ("iou", iou), ("acc", acc)):
                totals[key] += value
            if train and epoch == 0 and batch_index == 0:
                probs = torch.sigmoid(logits)
                print(
                    "[Sanity] target-positive=", masks.mean().item(),
                    "prob min/mean/max=", probs.min().item(), probs.mean().item(), probs.max().item(),
                    "predicted-positive=", (probs >= 0.5).float().mean().item(),
                )
        return {key: value / len(loader) for key, value in totals.items()}

    def _save(self, epoch, tag):
        torch.save({
            "epoch": epoch, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "best_dice": self.best_val_dice,
            "history": self.history, "cfg": vars(self.cfg),
        }, self.checkpoint_dir / f"ckpt_{tag}.pt")

    def _load_best(self):
        state = torch.load(self.checkpoint_dir / "ckpt_best.pt", map_location=self.device)
        self.model.load_state_dict(state["model"])

    def train(self):
        for epoch in range(self.cfg.epochs):
            self.scheduler.step(epoch)
            started = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader)
            record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            self.history.append(record)
            print(
                f"Epoch {epoch + 1:03d}/{self.cfg.epochs} ({time.time() - started:.1f}s)\n"
                f"  Train loss={train_metrics['loss']:.4f} bce={train_metrics['bce']:.4f} "
                f"mse={train_metrics['mse']:.4f} dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f}\n"
                f"  Val   loss={val_metrics['loss']:.4f} bce={val_metrics['bce']:.4f} "
                f"mse={val_metrics['mse']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f}"
            )
            self._save(epoch, "last")
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self._save(epoch, "best")
            with open(self.checkpoint_dir / "training_log.json", "w") as file:
                json.dump(self.history, file, indent=2)

    @torch.no_grad()
    def evaluate_test(self):
        self._load_best()
        metrics = self._run_epoch(self.test_loader)
        print(f"[Test] Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} Acc={metrics['acc']:.4f}")
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Dental caries ViT segmentation")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--pos_weight", type=float, default=20.0)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    trainer = Trainer(parse_args())
    trainer.train()
    trainer.evaluate_test()
