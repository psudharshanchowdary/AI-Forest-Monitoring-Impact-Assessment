from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 7 Hugging Face deployment helper")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "yolov8_seg" / "weights" / "best.pt")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--space-repo-id", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "deployment")
    parser.add_argument("--token-env", type=str, default="HF_TOKEN")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def write_model_card(output_dir: Path, repo_id: str, checkpoint_name: str, metrics: dict[str, float] | None) -> None:
    metrics_section = "No evaluation metrics file was found during packaging."
    if metrics:
        metrics_section = "\n".join([f"- {key}: {value}" for key, value in metrics.items()])
    output_dir.joinpath("README.md").write_text(
        "\n".join([
            f"# {repo_id}",
            "",
            "YOLOv8 segmentation model for forest, field, lake, road, and building detection from satellite imagery.",
            "",
            "## Files",
            f"- `{checkpoint_name}`: trained model checkpoint",
            "- `hf_inference.py`: reusable inference helper",
            "",
            "## Evaluation",
            metrics_section,
        ]),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in env var {args.token_env}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model_repo"
    space_dir = args.output_dir / "space_repo"
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.checkpoint, model_dir / args.checkpoint.name)
    shutil.copy2(ROOT / "deployment" / "hf_inference.py", model_dir / "hf_inference.py")

    metrics_path = ROOT / "outputs" / "evaluation" / "yolo_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    write_model_card(model_dir, repo_id=args.repo_id, checkpoint_name=args.checkpoint.name, metrics=metrics)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo_id, repo_type="model")

    if args.space_repo_id:
        space_dir.mkdir(parents=True, exist_ok=True)
        for name in ["app.py", "hf_inference.py", "requirements.txt", "README.md"]:
            shutil.copy2(ROOT / "deployment" / name, space_dir / name)
        readme_path = space_dir / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        content = content.replace("__MODEL_REPO_ID__", args.repo_id)
        readme_path.write_text(content, encoding="utf-8")
        app_path = space_dir / "app.py"
        content = app_path.read_text(encoding="utf-8")
        content = content.replace("__MODEL_REPO_ID__", args.repo_id)
        app_path.write_text(content, encoding="utf-8")
        api.create_repo(repo_id=args.space_repo_id, private=args.private, exist_ok=True, repo_type="space", space_sdk="gradio")
        api.upload_folder(folder_path=str(space_dir), repo_id=args.space_repo_id, repo_type="space")

    print(json.dumps({
        "model_repo": args.repo_id,
        "space_repo": args.space_repo_id or None,
        "packaged_checkpoint": str(model_dir / args.checkpoint.name),
    }, indent=2))


if __name__ == "__main__":
    main()
