import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import get_dataloaders
from src.evaluate import evaluate
from src.models import ResNet18, get_resnet50, get_efficientnet_b0


def get_model(name: str, num_classes: int):
    if name == "resnet18_scratch":
        return ResNet18(num_classes=num_classes)
    if name == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=True)
    if name == "efficientnet":
        return get_efficientnet_b0(num_classes=num_classes, pretrained=True)
    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "fc" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


DEFAULTS = {
    "model": "resnet50",
    "metadata": "data/metadata.json",
    "output_dir": "outputs",
    "batch_size": 64,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "image_size": 224,
    "num_workers": 4,
    "gpus": "0,1",
    "finetune_epochs": 5,
}


def run(
    model: str = "resnet50",
    metadata: str = "data/metadata.json",
    output_dir: str = "outputs",
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    num_workers: int = 4,
    gpus: str = "0,1",
    finetune_epochs: int = 5,
) -> dict:
    device_ids = [int(x) for x in gpus.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        metadata_path=metadata,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    net = get_model(model, num_classes)
    if torch.cuda.is_available() and len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = ckpt_dir / f"{model}_best.pt"
    ckpt_last = ckpt_dir / f"{model}_last.pt"
    results_path = Path(output_dir) / f"{model}_results.json"

    best_f1 = 0.0
    history = []
    start_epoch = 0
    pretrained = model in ("resnet50", "efficientnet")

    if ckpt_last.exists():
        ckpt = torch.load(ckpt_last, map_location=device, weights_only=False)
        if hasattr(net, "module"):
            net.module.load_state_dict(ckpt["state_dict"])
        else:
            net.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                prev = json.load(f)
                history = prev.get("history", [])
        epochs_phase2 = epochs
        print(f"Resuming from epoch {start_epoch}, best_val_f1={best_f1:.4f} (ещё {epochs} эпох)")
    else:
        if pretrained:
            freeze_backbone(net.module if hasattr(net, "module") else net)
            head_params = [p for p in net.parameters() if p.requires_grad]
            optimizer = AdamW(head_params, lr=lr, weight_decay=weight_decay)
            print("Phase 1: training classifier head only")
            for ep in range(finetune_epochs):
                train_loss = train_epoch(net, train_loader, criterion, optimizer, device)
                acc, f1, _ = evaluate(net, val_loader, device)
                history.append({
                    "epoch": ep + 1,
                    "phase": "head",
                    "train_loss": float(train_loss),
                    "val_acc": float(acc),
                    "val_f1": float(f1),
                })
                best_f1 = max(best_f1, f1)
                print(f"  Epoch {ep+1}/{finetune_epochs} Loss: {train_loss:.4f} Val Acc: {acc:.4f} F1: {f1:.4f}")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": model,
                        "test_accuracy": acc,
                        "test_f1_macro": best_f1,
                        "best_val_f1": best_f1,
                        "history": history,
                    }, f, indent=2, ensure_ascii=False)
            unfreeze_all(net.module if hasattr(net, "module") else net)
            optimizer = AdamW(net.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
            print("Phase 2: fine-tuning full model")
            start_epoch = 0
        epochs_phase2 = epochs - (finetune_epochs if pretrained else 0)

    for ep in range(epochs_phase2):
        global_epoch = start_epoch + ep
        train_loss = train_epoch(net, train_loader, criterion, optimizer, device)
        acc, f1, _ = evaluate(net, val_loader, device)
        scheduler.step(f1)
        history.append({
            "epoch": global_epoch + 1,
            "phase": "full",
            "train_loss": float(train_loss),
            "val_acc": float(acc),
            "val_f1": float(f1),
        })
        print(f"Epoch {global_epoch + 1} Loss: {train_loss:.4f} Val Acc: {acc:.4f} F1_macro: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            state = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            torch.save({"state_dict": state, "best_f1": best_f1, "epoch": global_epoch}, ckpt_best)
            print(f"  -> Saved best (F1={best_f1:.4f})")

        torch.save({
            "state_dict": (net.module if hasattr(net, "module") else net).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": global_epoch,
            "best_f1": best_f1,
        }, ckpt_last)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model,
                "test_accuracy": acc,
                "test_f1_macro": best_f1,
                "best_val_f1": best_f1,
                "history": history,
            }, f, indent=2, ensure_ascii=False)

    ckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
    if hasattr(net, "module"):
        net.module.load_state_dict(ckpt["state_dict"])
    else:
        net.load_state_dict(ckpt["state_dict"])

    test_acc, test_f1, _ = evaluate(net, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1_macro: {test_f1:.4f}")

    results = {
        "model": model,
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "best_val_f1": best_f1,
        "history": history,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_path}")
    return results
