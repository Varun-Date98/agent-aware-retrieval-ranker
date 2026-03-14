"""Fine-tune the cross-encoder reranker with agent-aware loss.

Run:  python -m training.train_reranker
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.utils import load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class PreTokenizedDataset(Dataset):
    """Loads pre-tokenized .pt files produced by prepare_data.py."""

    def __init__(self, pt_path: Path):
        data = torch.load(pt_path, weights_only=True)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]
        self.passage_tokens = data["passage_tokens"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "passage_tokens": self.passage_tokens[idx],
        }


def agent_aware_loss(logits: torch.Tensor, labels: torch.Tensor,
                     passage_tokens: torch.Tensor) -> torch.Tensor:
    """BCE loss with token-efficiency weighting on positives.

    Positive labels are scaled by 1/log(token_count) so shorter,
    information-dense passages get a stronger gradient signal.
    """
    probs = torch.sigmoid(logits)
    token_weight = 1.0 / torch.log(passage_tokens.clamp(min=2.0))
    # Weight positives by token efficiency, negatives get weight 1.0
    weight = torch.where(labels > 0.5, token_weight, torch.ones_like(token_weight))
    bce = -(labels * torch.log(probs + 1e-8) + (1 - labels) * torch.log(1 - probs + 1e-8))
    return (weight * bce).mean()


def train():
    cfg = load_config()
    tcfg = cfg["training"]
    rcfg = cfg["reranker"]

    device = torch.device(tcfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        rcfg["model"], num_labels=1
    )
    model.to(device)

    train_ds = PreTokenizedDataset(DATA_DIR / "train.pt")
    val_ds = PreTokenizedDataset(DATA_DIR / "val.pt")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=tcfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=tcfg["pin_memory"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    total_steps = len(train_loader) * tcfg["epochs"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = tcfg["mixed_precision"] and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    accum_steps = tcfg["gradient_accumulation_steps"]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(tcfg["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['epochs']}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            passage_tokens = batch["passage_tokens"].to(device)

            with autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = agent_aware_loss(logits, labels, passage_tokens)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                passage_tokens = batch["passage_tokens"].to(device)

                with autocast("cuda", enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    loss = agent_aware_loss(logits, labels, passage_tokens)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)

        avg_val = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  val_acc={val_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = CKPT_DIR / "reranker_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint -> {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()
