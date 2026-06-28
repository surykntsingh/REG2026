"""
Interface 0 — Visual Grounding (template implementation).

This is where you plug in your VQA / vision-language model.

Steps:
  1. Load your model weights (MODEL_PATH is available from core.py).
  2. Replace the body of predict_visual_context_response with your inference logic.
  3. If your model lives in a separate repository, add it under src/ and import it here:

       from src.your_repo.your_module import YourVQAModel

Platform rules for this interface:
  - Return a clinically grounded free-text answer.
  - The output is a plain JSON string — just the answer text, nothing else.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from core import load_json_file, load_roi_image
from path_wsi_reasoner.main_metric_b import predict_metric_b_single_case_from_model_dir


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    """
    Run Visual Grounding inference for a single ROI.

    Args:
        question_path:  Path to visual-context-question.json (platform-fixed).
        roi_image_path: Path to histopathology-region-of-interest-thumbnail.jpeg.

    Returns:
        A plain string answer (written as a JSON string by inference.py).

    IMPORTANT — do not change the return type (must stay ``str``). The platform
    expects a plain JSON string in visual-context-response.json; changing this
    will break submission validation.
    """
    question: str = load_json_file(location=question_path)
    roi_image = load_roi_image(location=roi_image_path)

    try:
        return predict_metric_b_single_case_from_model_dir(
            question=question,
            roi_image=roi_image,
            model_dir=Path("/opt/ml/model"),
        )
    except Exception:
        print("[interf0] ERROR: Metric B inference failed.", flush=True)
        traceback.print_exc()
        return "Unable to generate a visual-context response for this case."
