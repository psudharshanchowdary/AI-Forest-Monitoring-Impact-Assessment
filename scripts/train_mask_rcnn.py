from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forest_monitor.models.evaluation import evaluate_mask_rcnn, save_confusion_matrix, save_training_plot, write_metrics
from forest_monitor.models.mask_rcnn_dataset import YoloSegmentationInstanceDataset, build_mask_rcnn_model, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 Mask R-CNN training")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mask_rcnn")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()
        total_loss += float(losses.item())
        batches += 1
    return total_loss / max(batches, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = YoloSegmentationInstanceDataset(args.dataset_root, split="train")
    val_dataset = YoloSegmentationInstanceDataset(args.dataset_root, split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_mask_rcnn_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history: list[dict[str, float]] = []
    best_map = -1.0
    best_checkpoint = args.output_dir / "best.pt"
    last_checkpoint = args.output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics, _ = evaluate_mask_rcnn(model, val_loader, device=device)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_map_50": val_metrics["map50"],
            "val_map_50_95": val_metrics["map50_95"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, last_checkpoint)
        if val_metrics["map50_95"] > best_map:
            best_map = val_metrics["map50_95"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, best_checkpoint)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_map={val_metrics['map50_95']:.4f} val_iou={val_metrics['iou']:.4f}")

    metrics, confusion = evaluate_mask_rcnn(model, val_loader, device=device)
    metrics.update({
        "model": "Mask R-CNN",
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
    })
    write_metrics(metrics, args.output_dir / "metrics.json")
    save_confusion_matrix(confusion, args.output_dir / "confusion_matrix.png", title="Mask R-CNN Confusion Matrix")
    save_training_plot(history, args.output_dir / "results.png")

    with (args.output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
