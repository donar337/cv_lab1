import argparse
import json
from pathlib import Path
from collections import Counter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_samples(root: Path) -> list[tuple[str, str]]:
    samples = []
    root = Path(root).resolve()

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = img_path.stem.split("$$")
        if len(parts) < 4:
            continue
        colour = parts[3]
        samples.append((str(img_path), colour))

    return samples


def run(
    data_root: str | Path = "data/dvm_front",
    output: str | Path = "data/metadata.json",
    min_samples: int = 10,
):
    root = Path(data_root)
    if not root.exists():
        print(f"Error: {root} does not exist. Run scripts/download_data.py first.")
        return 1

    samples = collect_samples(root)
    colour_counts = Counter(s for _, s in samples)

    # Filter classes with too few samples
    filtered = [(p, c) for p, c in samples if colour_counts[c] >= min_samples]
    if len(filtered) < len(samples):
        removed = len(samples) - len(filtered)
        print(f"Filtered {removed} samples from rare classes (< {min_samples})")

    samples = filtered
    colours = sorted(set(c for _, c in samples))
    colour_to_idx = {c: i for i, c in enumerate(colours)}

    metadata = {
        "samples": [{"path": p, "label": colour_to_idx[c]} for p, c in samples],
        "label_to_class": colours,
        "class_to_label": colour_to_idx,
        "num_classes": len(colours),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata to {out_path}")
    print(f"Total samples: {len(samples)}, classes: {len(colours)}")
