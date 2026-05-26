from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from forest_monitor.experiments.ablation import run_ablation_study
METRICS = ["map50", "map50_95", "iou", "precision", "recall", "f1"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 ablation study")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--yolo-checkpoint", type=Path, default=ROOT / "outputs" / "yolov8_seg" / "weights" / "best.pt")
    parser.add_argument("--maskrcnn-checkpoint", type=Path, default=ROOT / "outputs" / "mask_rcnn" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ablation")
    parser.add_argument("--patch-size", type=int, default=640)
    return parser.parse_args()
def save_plots(table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_table = table.melt(id_vars=["variant"], value_vars=METRICS, var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_table, x="metric", y="value", hue="variant")
    plt.title("Ablation Metrics Across Model Variants")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_metric_bars.png", dpi=180)
    plt.close()
    baseline = table.set_index("variant").loc["Full model"]
    delta_rows = []
    for _, row in table.iterrows():
        if row["variant"] == "Full model":
            continue
        for metric in ["map50_95", "iou", "precision", "recall"]:
            delta_rows.append({
                "variant": row["variant"],
                "metric": metric,
                "drop_percent": ((baseline[metric] - row[metric]) / max(baseline[metric], 1e-6)) * 100.0,
            })
    delta_df = pd.DataFrame(delta_rows)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=delta_df, x="variant", y="drop_percent", hue="metric")
    plt.axhline(0.0, color="#94a3b8", linewidth=1.0)
    plt.title("Performance Drop Relative to Full Model")
    plt.ylabel("Drop (%)")
    plt.xlabel("Model Variant")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_performance_drop.png", dpi=180)
    plt.close()
    f1_columns = [column for column in table.columns if column.startswith("f1_")]
    f1_table = table[["variant", *f1_columns]].melt(id_vars=["variant"], var_name="class_name", value_name="f1")
    f1_table["class_name"] = f1_table["class_name"].str.replace("f1_", "", regex=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=f1_table, x="class_name", y="f1", hue="variant")
    plt.title("Per-Class F1 Score by Ablation Variant")
    plt.xlabel("Class")
    plt.ylabel("F1 score")
    plt.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_f1_scores.png", dpi=180)
    plt.close()
def main() -> None:
    args = parse_args()
    table, conclusions = run_ablation_study(
        dataset_root=args.dataset_root,
        yolo_checkpoint=args.yolo_checkpoint,
        maskrcnn_checkpoint=args.maskrcnn_checkpoint,
        output_dir=args.output_dir,
        patch_size_px=args.patch_size,
    )
    save_plots(table, args.output_dir)
    display_columns = ["variant", "map50", "map50_95", "iou", "precision", "recall", "f1"]
    print(table[display_columns].to_string(index=False))
    print("\nConclusions")
    for line in conclusions:
        print(f"- {line}")
if __name__ == "__main__":
    main()
