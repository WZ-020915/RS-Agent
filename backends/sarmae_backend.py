from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MMROTATE_ROOT = Path("/root/autodl-tmp/mmrotate")
SARMAE_ROOT = Path("/root/autodl-tmp/SARMAE-main")
SEG_ROOT = SARMAE_ROOT / "SARMAE_Fintune" / "Segmentation"
MMSEG_ROOT = SARMAE_ROOT / "mmsegmentation"

DETECT_CONFIG = MMROTATE_ROOT / "configs" / "SARMAE" / "SSDD" / "vitb_ssdd_local.py"
DETECT_CKPT = SARMAE_ROOT / "weights" / "detect_epoch_34.pth"
SEG_CONFIG = SEG_ROOT / "work_dirs" / "vit-b-airseg-polar-20260525_132443" / "vit-b-airseg-polar-20260525.py"
SEG_CKPT = SARMAE_ROOT / "weights" / "seg_iter_20000.pth"


def summarize_detection_result(result, score_thr: float):
    detections = []
    bbox_result = result[0] if isinstance(result, tuple) else result
    for class_id, class_result in enumerate(bbox_result):
        arr = np.asarray(class_result)
        if arr.size == 0:
            continue
        for row in arr:
            score = float(row[-1])
            if score < score_thr:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "score": score,
                    "bbox": [float(x) for x in row[:-1].tolist()],
                }
            )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def run_detection(req: dict) -> dict:
    sys.path.insert(0, str(MMROTATE_ROOT))
    import mmcv_custom  # noqa: F401
    import mmrotate  # noqa: F401
    from mmdet.apis import inference_detector, init_detector, show_result_pyplot

    image_path = req["image_path"]
    score_thr = float(req.get("score_thr") or 0.3)
    output_dir = Path(req["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / (Path(image_path).stem + "_sarmae_detect.jpg")
    raw_file = output_dir / (Path(image_path).stem + "_sarmae_detect.json")

    model = init_detector(str(DETECT_CONFIG), str(DETECT_CKPT), device=req.get("device") or "cuda:0")
    result = inference_detector(model, image_path)
    show_result_pyplot(model, image_path, result, palette="random", score_thr=score_thr, out_file=str(out_file))
    detections = summarize_detection_result(result, score_thr)
    payload = {
        "model": "SARMAE",
        "task": "detect",
        "image": image_path,
        "config": str(DETECT_CONFIG),
        "checkpoint": str(DETECT_CKPT),
        "score_thr": score_thr,
        "num_detections": len(detections),
        "detections": detections,
        "visualization": str(out_file),
        "raw_result": str(raw_file),
    }
    raw_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def colorize(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [255, 210, 0],
            [0, 170, 255],
            [30, 220, 120],
            [255, 80, 80],
            [180, 80, 255],
            [255, 150, 40],
        ],
        dtype=np.uint8,
    )
    if int(mask.max()) >= len(palette):
        extra = int(mask.max()) + 1 - len(palette)
        rng = np.random.default_rng(0)
        palette = np.concatenate([palette, rng.integers(0, 255, size=(extra, 3), dtype=np.uint8)], axis=0)
    return palette[mask.astype(np.int64)]


def run_segmentation(req: dict) -> dict:
    sys.path.insert(0, str(SEG_ROOT))
    sys.path.insert(0, str(MMSEG_ROOT))
    try:
        import mmcv_custom  # noqa: F401
        from mmengine.model import revert_sync_batchnorm
        from mmseg.apis import inference_model, init_model, show_result_pyplot
    except Exception as exc:
        raise RuntimeError(
            "SARMAE segmentation requires the mmsegmentation/mmengine stack. "
            "The current 'sarmae' env was configured for mmrotate detection "
            "with mmcv-full==1.6.1, which is incompatible with mmseg>=1.x. "
            "Use a separate SARMAE segmentation env with mmcv>=2, or restore "
            "the original segmentation environment before calling task='segment'."
        ) from exc

    image_path = req["image_path"]
    output_dir = Path(req["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / (Path(image_path).stem + "_sarmae_segment.png")
    raw_file = output_dir / (Path(image_path).stem + "_sarmae_segment_mask.png")
    json_file = output_dir / (Path(image_path).stem + "_sarmae_segment.json")

    model = init_model(str(SEG_CONFIG), str(SEG_CKPT), device=req.get("device") or "cuda:0")
    if (req.get("device") or "").startswith("cpu"):
        model = revert_sync_batchnorm(model)
    result = inference_model(model, image_path)
    show_result_pyplot(model, image_path, result, show=False, out_file=str(out_file), opacity=0.5, with_labels=False)
    pred = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(pred).save(raw_file)
    Image.fromarray(colorize(pred)).save(output_dir / (Path(image_path).stem + "_sarmae_segment_color.png"))
    classes, counts = np.unique(pred, return_counts=True)
    payload = {
        "model": "SARMAE",
        "task": "segment",
        "image": image_path,
        "config": str(SEG_CONFIG),
        "checkpoint": str(SEG_CKPT),
        "mask": str(raw_file),
        "visualization": str(out_file),
        "class_pixel_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }
    json_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())
    task = req.get("task")
    if task == "detect":
        result = run_detection(req)
    elif task == "segment":
        result = run_segmentation(req)
    else:
        raise ValueError("SARMAE task must be 'detect' or 'segment'")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
