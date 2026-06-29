"""
Interface 1 — Workflow Reasoning (template implementation).

This is where you plug in your chain-of-thought / reasoning model.

Steps:
  1. Load your model weights (MODEL_PATH is available from core.py).
  2. Replace the body of predict_chain_of_thought with your inference logic.
  3. If your model lives in a separate repository, add it under src/ and import it here:

       from src.your_repo.your_module import YourReasoningModel

Platform rules for this interface:
  - Use the EXACT canonical question / next_question strings from the training
    annotations. Minor whitespace/capitalisation differences are normalised, but
    paraphrasing is penalised.
  - Output is a bare JSON array of steps — do NOT wrap it in an object with an
    "id" key. The platform adds that wrapper automatically.
  - The last step must have "next_question": "".

Output schema per step:
  {
    "question":      "<exact canonical question string>",
    "answer":        "<your model's free-text answer>",
    "next_question": "<exact canonical next question string, or empty string>"
  }
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import TypedDict

import torch

from core import MODEL_PATH
from path_wsi_reasoner.main import predict_metric_a_single_case
from trident import extract_conch_v15_features_for_wsi


def _log_stage(message: str, start_time: float | None = None) -> float:
    now = time.monotonic()
    if start_time is None:
        print(f"[interf1] {message}", flush=True)
    else:
        print(f"[interf1] {message} ({now - start_time:.2f}s)", flush=True)
    return now


class ChainOfThoughtStep(TypedDict):
    """One reasoning step — keys must match the platform schema exactly."""

    question: str
    answer: str
    next_question: str


def _require_model_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found at expected path: {path}")
    return path


def _resolve_conch_v15_weights_path() -> Path:
    return _require_model_file(
        MODEL_PATH / "conch_v15" / "pytorch_model_vision.bin",
        "CONCH v1.5 checkpoint",
    )


def _prepare_trident_offline_weights() -> None:
    os.environ["TRIDENT_SEG_HEST_WEIGHTS"] = str(
        _require_model_file(
            MODEL_PATH / "hest" / "deeplabv3_seg_v4.ckpt",
            "HEST segmenter checkpoint",
        )
    )
    os.environ["TRIDENT_SEG_GRANDQC_ARTIFACT_WEIGHTS"] = str(
        _require_model_file(
            MODEL_PATH / "grandqc_artifact" / "GrandQC_MPP1_state_dict.pth",
            "GrandQC artifact checkpoint",
        )
    )


def _resolve_metric_a_config_path() -> Path:
    return _require_model_file(
        MODEL_PATH / "config.yaml",
        "Metric A config.yaml",
    )


def _resolve_metric_a_checkpoint_path() -> Path:
    return _require_model_file(
        MODEL_PATH / "cot_workflow_model.ckpt",
        "Metric A checkpoint",
    )


def _resolve_metric_a_reports_path() -> Path:
    return _require_model_file(
        MODEL_PATH / "train_CoT_v01.json",
        "Metric A reports JSON",
    )


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """
    Run Workflow Reasoning inference for a single whole-slide image.

    Args:
        wsi_path: Path to /input/images/whole-slide-image/<uid>.tiff
            (<uid> is an opaque UUID hash from inputs.json image.name).

    Returns:
        A list of steps, each with keys: question, answer, next_question.
        Use exact canonical question strings from the training annotations.

    IMPORTANT — do not change the return type (must stay
    ``list[ChainOfThoughtStep]``). inference.py serialises this list directly
    to chain-of-thought.json; wrapping it or changing field names will break
    submission validation.
    """
    try:
        total_start = _log_stage("starting Metric A inference")
        _prepare_trident_offline_weights()
        stage_start = _log_stage("generating CONCH features with TRIDENT")
        feature_path = extract_conch_v15_features_for_wsi(
            wsi_path=wsi_path,
            job_dir=Path("/tmp/reg2026_trident"),
            patch_encoder_weights_path=_resolve_conch_v15_weights_path(),
            segmenter="hest",
            seg_conf_thresh=0.1,
            mag=20,
            patch_size=512,
            batch_size=64,
            segmentation_batch_size=64,
            dataloader_workers=0,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            mpp=0.5,
            reader_type="openslide",
            reader_type_fallbacks=(),
            feature_reader_type="tiffslide",
            fallback_segmenters=False,
            remove_artifacts=False,
            remove_holes=False,
            reuse_existing=True,
        )
        stage_start = _log_stage(f"generated CONCH features at {feature_path}", stage_start)

        stage_start = _log_stage("running Metric A single-case prediction")
        prediction = predict_metric_a_single_case(
            wsi_path=wsi_path,
            config_file_path=_resolve_metric_a_config_path(),
            checkpoint_path=_resolve_metric_a_checkpoint_path(),
            reports_json_path=_resolve_metric_a_reports_path(),
            feature_path=feature_path,
        )
        _log_stage("completed Metric A single-case prediction", stage_start)
        _log_stage("completed Metric A inference", total_start)
        return prediction
    except Exception:
        print(f"[interf1] ERROR: Metric A inference failed for {wsi_path}.", flush=True)
        traceback.print_exc()
        return [
            {
                "question": "What is the final pathology report?",
                "answer": "Unable to generate a pathology report for this case.",
                "next_question": "",
            }
        ]
