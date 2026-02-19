import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DVMCarColorDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"

        with open(metadata_path) as f:
            meta = json.load(f)

        self.samples = meta["samples"]
        self.label_to_class = meta["label_to_class"]
        self.num_classes = meta["num_classes"]

        # Stratified split
        rng = torch.Generator().manual_seed(seed)
        n = len(self.samples)
        indices = torch.randperm(n, generator=rng)

        t_end = int(n * train_ratio)
        v_end = t_end + int(n * val_ratio)

        if split == "train":
            self.indices = indices[:t_end].tolist()
        elif split == "val":
            self.indices = indices[t_end:v_end].tolist()
        else:
            self.indices = indices[v_end:].tolist()

        self.transform = self._build_transform()

    def _build_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.image_size + 32, self.image_size + 32)),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        i = self.indices[idx]
        sample = self.samples[i]
        path = sample["path"]
        label = sample["label"]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def get_dataloaders(
    metadata_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    train_ds = DVMCarColorDataset(
        metadata_path,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        image_size=image_size,
        augment=True,
    )
    val_ds = DVMCarColorDataset(
        metadata_path,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        image_size=image_size,
        augment=False,
    )
    test_ds = DVMCarColorDataset(
        metadata_path,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        image_size=image_size,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds.num_classes
