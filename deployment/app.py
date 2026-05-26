from __future__ import annotations

import os

import gradio as gr
import numpy as np
import pandas as pd

from hf_inference import ForestSegmentationInference

MODEL_REPO_ID = os.environ.get("HF_MODEL_ID", "__MODEL_REPO_ID__")
segmenter = ForestSegmentationInference(model_repo_id=MODEL_REPO_ID)


def run_inference(image: np.ndarray):
    overlay, records = segmenter.predict(image)
    return overlay, pd.DataFrame(records)


with gr.Blocks(title="Forest Monitoring Segmentation Demo") as demo:
    gr.Markdown("# AI-Powered Forest Monitoring System")
    gr.Markdown("Upload a satellite image to view YOLOv8 segmentation results.")
    with gr.Row():
        image_input = gr.Image(label="Input image", type="numpy")
        image_output = gr.Image(label="Segmented output", type="numpy")
    table_output = gr.Dataframe(label="Detected instances")
    run_button = gr.Button("Run Inference")
    run_button.click(fn=run_inference, inputs=image_input, outputs=[image_output, table_output])


demo.launch()
