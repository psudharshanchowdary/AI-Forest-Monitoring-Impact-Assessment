# Forest Monitoring Gradio Space

This Hugging Face Space loads the segmentation checkpoint from `__MODEL_REPO_ID__` and runs land-cover segmentation on uploaded satellite images.

## Files
- `app.py`: Gradio app entrypoint
- `hf_inference.py`: shared YOLOv8 inference helper
- `requirements.txt`: Space dependencies
