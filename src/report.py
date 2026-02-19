import json
from pathlib import Path

MODEL_NAMES = ["resnet18_scratch", "resnet50", "efficientnet"]


def load_results(output_dir: str | Path = "outputs") -> list[dict]:
    output_dir = Path(output_dir)
    results = []

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)
    else:
        for name in MODEL_NAMES:
            p = output_dir / f"{name}_results.json"
            if p.exists():
                with open(p) as f:
                    results.append(json.load(f))
        if results:
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)

    return sorted(results, key=lambda x: x["test_f1_macro"], reverse=True)


def format_report(results: list[dict]) -> str:
    if not results:
        return "_Нет результатов. Запустите обучение._"

    lines = [
        "## Сравнение классификаторов",
        "",
        "| Модель | Accuracy | F1_macro | Epochs |",
        "|--------|----------|----------|--------|",
    ]
    for r in results:
        lines.append(f"| {r['model']} | {r['test_accuracy']:.4f} | {r['test_f1_macro']:.4f} | {len(r['history'])} |")
    return "\n".join(lines)


def _get_head_finetune_boundary(hist: list) -> int | None:
    head_epochs = [h["epoch"] for h in hist if h.get("phase") == "head"]
    return max(head_epochs) if head_epochs else None


def _build_epochs_values(hist: list, value_key: str) -> tuple[list[int], list[float]]:
    boundary = _get_head_finetune_boundary(hist)
    epochs, values = [], []
    for h in hist:
        if value_key not in h:
            continue
        if h.get("phase") == "head":
            epochs.append(h["epoch"])
        else:
            # full: эпохи в JSON идут 1,2,3... — продолжаем после boundary
            base = (boundary or 0) + 1
            epochs.append(base + (h["epoch"] - 1))
        values.append(h[value_key])
    return epochs, values


def plot_f1_vs_epochs(results: list[dict], ax=None) -> None:
    import matplotlib.pyplot as plt

    if not results:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"resnet18_scratch": "#3498db", "resnet50": "#2ecc71", "efficientnet": "#e74c3c"}
    max_head_epoch = 0
    for r in results:
        hist = r.get("history", [])
        if not hist:
            continue
        boundary = _get_head_finetune_boundary(hist)
        if boundary is not None:
            max_head_epoch = max(max_head_epoch, boundary)
        epochs, f1s = _build_epochs_values(hist, "val_f1")
        if epochs and f1s:
            ax.plot(epochs, f1s, label=r["model"], color=colors.get(r["model"], None), linewidth=2)

    if max_head_epoch > 0:
        ax.axvline(max_head_epoch + 0.5, color="gray", linestyle=":", alpha=0.8, linewidth=1.5)
        ax.axvspan(0.5, max_head_epoch + 0.5, alpha=0.08, color="gray")
        x_right = max(ax.get_xlim()[1], max_head_epoch + 5)
        ax.text(max_head_epoch / 2 + 0.5, 0.95, "head", ha="center", fontsize=9, alpha=0.7)
        ax.text((max_head_epoch + 1 + x_right) / 2, 0.95, "finetune", ha="center", fontsize=9, alpha=0.7)

    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.7, label="F1=0.8")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("F1_macro (val)")
    ax.set_title("F1_macro от количества эпох")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss_vs_epochs(results: list[dict], ax=None) -> None:
    import matplotlib.pyplot as plt

    if not results:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"resnet18_scratch": "#3498db", "resnet50": "#2ecc71", "efficientnet": "#e74c3c"}
    max_head_epoch = 0
    for r in results:
        hist = r.get("history", [])
        if not hist:
            continue
        boundary = _get_head_finetune_boundary(hist)
        if boundary is not None:
            max_head_epoch = max(max_head_epoch, boundary)
        epochs, losses = _build_epochs_values(hist, "train_loss")
        if epochs and losses:
            ax.plot(epochs, losses, label=r["model"], color=colors.get(r["model"], None), linewidth=2)

    if max_head_epoch > 0:
        ax.axvline(max_head_epoch + 0.5, color="gray", linestyle=":", alpha=0.8, linewidth=1.5)
        ax.axvspan(0.5, max_head_epoch + 0.5, alpha=0.08, color="gray")
        y_top = ax.get_ylim()[1]
        x_right = ax.get_xlim()[1]
        ax.text(max_head_epoch / 2 + 0.5, y_top * 0.98, "head", ha="center", fontsize=9, alpha=0.7)
        ax.text((max_head_epoch + 1 + x_right) / 2, y_top * 0.98, "finetune", ha="center", fontsize=9, alpha=0.7)

    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Train Loss")
    ax.set_title("Функция потерь от количества эпох")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
