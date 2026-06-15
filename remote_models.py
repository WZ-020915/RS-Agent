"""Callable wrappers for the local remote-sensing models.

Each model lives in a different conda environment.  These functions keep the
public API in one Python module and run the model-specific backend in its own
environment, returning a JSON-serializable dict.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BACKENDS = ROOT / "backends"
CONDA_SH = Path("/root/miniconda3/etc/profile.d/conda.sh")


class ModelWrapperError(RuntimeError):
    """Raised when a backend process fails or returns invalid output."""


def _run_backend(env_name: str, script_name: str, payload: dict[str, Any], timeout: int | None) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        request_path = Path(tmp.name)

    script = BACKENDS / script_name
    cmd = (
        f"source {shlex.quote(str(CONDA_SH))} && "
        f"conda activate {shlex.quote(env_name)} && "
        f"OMP_NUM_THREADS=1 python {shlex.quote(str(script))} "
        f"--request {shlex.quote(str(request_path))}"
    )
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(ROOT.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    finally:
        request_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise ModelWrapperError(
            f"{script_name} failed in conda env {env_name!r} with exit code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )

    text = proc.stdout.strip()
    if not text:
        raise ModelWrapperError(f"{script_name} returned no JSON output. STDERR:\n{proc.stderr[-4000:]}")
    last_line = text.splitlines()[-1]
    try:
        return json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ModelWrapperError(
            f"{script_name} returned invalid JSON in the last stdout line: {last_line!r}\n"
            f"Full stdout tail:\n{text[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        ) from exc


def skyeyegpt(
    image_path: str | os.PathLike[str] | None = None,
    prompt: str = "",
    task: str = "chat",
    history: list[dict[str, str] | tuple[str, str]] | None = None,
    max_new_tokens: int = 512,
    min_new_tokens: int | None = None,
    temperature: float = 0.6,
    num_beams: int = 1,
    output_dir: str | os.PathLike[str] = "/root/autodl-tmp/model_wrappers/outputs/skyeyegpt",
    timeout: int | None = 600,
) -> dict[str, Any]:
    """Run SkyEyeGPT for chat, grounding, captioning, or VQA.

    Args:
        image_path: Optional image path. Required for caption/VQA/grounding.
        prompt: User question or instruction.
        task: One of ``chat``, ``caption``, ``vqa``, or ``grounding``.
        history: Optional prior turns as ``{"role": "user"|"assistant",
            "content": "..."}`` dicts or ``(role, content)`` tuples. The
            backend replays this history before answering the current prompt.
        max_new_tokens: Maximum number of generated tokens.
        min_new_tokens: Optional minimum number of generated tokens. Useful
            when captioning stops before a complete sentence.
        temperature: Sampling temperature.
        num_beams: Beam count used by generation.
        output_dir: Directory for any generated artifacts.

    Returns:
        Backend response as a dict. If the local SkyEyeGPT runtime is incomplete,
        the dict contains an actionable ``status="unavailable"`` reason.
    """

    payload = {
        "task": task,
        "image_path": str(image_path) if image_path is not None else None,
        "prompt": prompt,
        "history": history or [],
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "temperature": temperature,
        "num_beams": num_beams,
        "output_dir": str(output_dir),
    }
    return _run_backend("minigptv", "skyeyegpt_backend.py", payload, timeout)


def sarmae(
    image_path: str | os.PathLike[str],
    task: str = "detect",
    output_dir: str | os.PathLike[str] = "/root/autodl-tmp/model_wrappers/outputs/sarmae",
    score_thr: float = 0.3,
    device: str = "cuda:0",
    timeout: int | None = 900,
) -> dict[str, Any]:
    """Run SARMAE for SAR segmentation, target detection, or both.

    Args:
        image_path: Input image path.
        task: ``detect`` for SSDD target detection, ``segment`` for SAR semantic
            segmentation, or ``both`` to run detection and segmentation.
        output_dir: Directory where visualizations and raw results are written.
        score_thr: Detection score threshold.
        device: Torch device string.
    """

    payload = {
        "task": task,
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "score_thr": score_thr,
        "device": device,
    }
    if task == "detect":
        return _run_backend("sarmae", "sarmae_backend.py", payload, timeout)
    if task == "segment":
        return _run_backend("sarmae_seg", "sarmae_backend.py", payload, timeout)
    if task == "both":
        detect_payload = {**payload, "task": "detect", "output_dir": str(Path(output_dir) / "detect")}
        segment_payload = {**payload, "task": "segment", "output_dir": str(Path(output_dir) / "segment")}
        return {
            "model": "SARMAE",
            "task": "both",
            "image": str(image_path),
            "detect": _run_backend("sarmae", "sarmae_backend.py", detect_payload, timeout),
            "segment": _run_backend("sarmae_seg", "sarmae_backend.py", segment_payload, timeout),
        }
    raise ValueError("SARMAE task must be 'detect', 'segment', or 'both'")


def mtp(
    image_path: str | os.PathLike[str],
    task: str = "rotated",
    output_dir: str | os.PathLike[str] = "/root/autodl-tmp/model_wrappers/outputs/mtp",
    score_thr: float = 0.3,
    target_class: str | None = None,
    device: str = "cuda:0",
    timeout: int | None = 900,
) -> dict[str, Any]:
    """Run MTP DIOR object detection.

    Args:
        image_path: Input optical remote-sensing image path.
        task: ``horizontal``/``hbb`` for horizontal boxes, ``rotated``/``obb``
            for rotated boxes, or ``both`` to run both detectors.
        output_dir: Directory where visualization JPGs and JSON results are written.
        score_thr: Detection score threshold used for returned and drawn boxes.
        target_class: Optional DIOR class name or common Chinese alias. When set,
            returned detections and ``num_detections`` are filtered to that class.
        device: Torch device string.
    """

    payload = {
        "task": task,
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "score_thr": score_thr,
        "target_class": target_class,
        "device": device,
    }
    return _run_backend("mtp", "mtp_backend.py", payload, timeout)


def dofa(
    dataset: str,
    sample_id: str | None = None,
    input_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] = "/root/autodl-tmp/model_wrappers/outputs/dofa",
    split: str = "test",
    crop_size: int = 224,
    stride: int = 112,
    batch_size: int = 16,
    device: str = "cuda",
    timeout: int | None = 900,
) -> dict[str, Any]:
    """Run a DOFA segmentation head from ``outputs/dofa_seg_rgb_ms``.

    Args:
        dataset: One of ``m-NeonTree``, ``m-SA-crop-type``, ``m-cashew-plant``,
            ``m-chesapeake``, ``m-nz-cattle``, or ``m-pv4ger-seg``.
        sample_id: Dataset sample id from the partition json. If omitted, the
            first sample from ``split`` is used.
        input_path: Optional explicit HDF5 sample path. It must use the same
            band layout/statistics as the selected dataset.
        output_dir: Directory for RGB, mask, overlay, and manifest outputs.
    """

    payload = {
        "dataset": dataset,
        "sample_id": sample_id,
        "input_path": str(input_path) if input_path is not None else None,
        "output_dir": str(output_dir),
        "split": split,
        "crop_size": crop_size,
        "stride": stride,
        "batch_size": batch_size,
        "device": device,
    }
    return _run_backend("dofa", "dofa_backend.py", payload, timeout)


def dehazeformer(
    image_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = "/root/autodl-tmp/model_wrappers/outputs/dehazeformer",
    model_name: str = "dehazeformer-b",
    exp: str = "reside6k",
    checkpoint: str | os.PathLike[str] | None = None,
    device: str = "cuda:0",
    timeout: int | None = 900,
) -> dict[str, Any]:
    """Run DehazeFormer single-image dehazing on an RGB hazy/cloudy image.

    Args:
        image_path: Input hazy/cloudy RGB image path.
        output_dir: Directory where the dehazed image and result JSON are written.
        model_name: DehazeFormer variant, e.g. ``dehazeformer-b``.
        exp: Weight subdirectory under ``DehazeFormer/saved_models`` or
            ``DehazeFormer/save_models``. Defaults to ``reside6k``.
        checkpoint: Optional explicit ``.pth`` checkpoint path.
        device: Torch device string.
    """

    payload = {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "exp": exp,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "device": device,
    }
    return _run_backend("base", "dehazeformer_backend.py", payload, timeout)


def sattxt(
    task: str,
    image_paths: list[str | os.PathLike[str]] | str | os.PathLike[str] | None = None,
    texts: list[str] | None = None,
    categories: list[str] | None = None,
    output_dir: str | os.PathLike[str] = "/root/autodl-tmp/model_wrappers/outputs/sattxt",
    device: str = "cuda:0",
    timeout: int | None = 1200,
) -> dict[str, Any]:
    """Run SATtxt zero-shot classification or image-text retrieval.

    Args:
        task: ``zero_shot_classification``, ``image_to_text``, ``text_to_image``,
            or ``retrieval``.
        image_paths: One image path or a list of image paths.
        texts: Candidate captions/text queries for retrieval.
        categories: Class names for zero-shot classification.
    """

    if isinstance(image_paths, (str, os.PathLike)):
        normalized_images = [str(image_paths)]
    elif image_paths is None:
        normalized_images = []
    else:
        normalized_images = [str(path) for path in image_paths]

    payload = {
        "task": task,
        "image_paths": normalized_images,
        "texts": texts or [],
        "categories": categories or [],
        "output_dir": str(output_dir),
        "device": device,
    }
    return _run_backend("sattxt", "sattxt_backend.py", payload, timeout)
