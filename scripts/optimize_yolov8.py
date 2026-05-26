from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 7 Optuna tuning for YOLOv8 segmentation")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s-seg.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="yolov8_seg_optuna")
    parser.add_argument("--storage", type=str, default=f"sqlite:///{(ROOT / 'outputs' / 'optuna' / 'yolov8_optuna.db').as_posix()}")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs" / "optuna" / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "optuna")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def save_plots(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    completed = df[df["state"] == "COMPLETE"].copy()
    if completed.empty:
        return

    completed.to_csv(output_dir / "trials.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(completed["number"], completed["value"], marker="o", linewidth=1.5)
    plt.title("Optuna Optimization History")
    plt.xlabel("Trial")
    plt.ylabel("mAP@50-95")
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_history.png", dpi=180)
    plt.close()

    importances = optuna.importance.get_param_importances(study)
    importance_df = pd.DataFrame({"parameter": list(importances.keys()), "importance": list(importances.values())})
    importance_df.to_csv(output_dir / "parameter_importance.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=importance_df, x="importance", y="parameter", orient="h")
    plt.title("Optuna Parameter Importance")
    plt.xlabel("Importance")
    plt.ylabel("Parameter")
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_importance.png", dpi=180)
    plt.close()



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        lr0 = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [4, 8, 12, 16])
        momentum = trial.suggest_float("momentum", 0.80, 0.97)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

        model = YOLO(args.model)
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            project=str(args.project),
            name=f"trial_{trial.number:03d}",
            exist_ok=True,
            plots=False,
            cache=True,
            seed=args.seed,
            device=0 if torch.cuda.is_available() else "cpu",
        )
        save_dir = Path(results.save_dir)
        best_path = save_dir / "weights" / "best.pt"
        metrics = YOLO(str(best_path)).val(data=str(args.data), split="val", imgsz=args.imgsz, verbose=False)
        score = float(metrics.seg.map)
        trial.set_user_attr("map50", float(metrics.seg.map50))
        trial.set_user_attr("precision", float(metrics.seg.mp))
        trial.set_user_attr("recall", float(metrics.seg.mr))
        trial.set_user_attr("save_dir", str(save_dir))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    best = {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    (args.output_dir / "best_trial.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    save_plots(study, args.output_dir)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
