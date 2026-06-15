from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MTP_ROOT = Path("/root/autodl-tmp/MTP-main")
HORIZONTAL_ROOT = MTP_ROOT / "RS_Tasks_Finetune" / "Horizontal_Detection"
ROTATED_ROOT = MTP_ROOT / "RS_Tasks_Finetune" / "Rotated_Detection" / "mmrotate1.x"
WEIGHTS_ROOT = MTP_ROOT / "weights"

HORIZONTAL_CONFIG = HORIZONTAL_ROOT / "configs" / "mtp" / "dior" / "faster_rcnn_rvsa_b_800_mae_mtp_dior.py"
HORIZONTAL_CKPT = WEIGHTS_ROOT / "dior-rvsa-b-mae-mtp-epoch_12.pth"
ROTATED_CONFIG = ROTATED_ROOT / "configs" / "mtp" / "dior-r" / "oriented_rcnn_rvsa_b_800_mae_mtp_diorr.py"
ROTATED_CKPT = WEIGHTS_ROOT / "diorr-rvsa-b-mae-mtp-epoch_12.pth"

DIOR_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "expressway-service-area",
    "expressway-toll-station",
    "dam",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]

CLASS_ALIASES = {
    "飞机": "airplane",
    "机场": "airport",
    "棒球场": "baseballfield",
    "篮球场": "basketballcourt",
    "桥": "bridge",
    "桥梁": "bridge",
    "烟囱": "chimney",
    "高速服务区": "expressway-service-area",
    "收费站": "expressway-toll-station",
    "大坝": "dam",
    "高尔夫球场": "golffield",
    "田径场": "groundtrackfield",
    "港口": "harbor",
    "立交桥": "overpass",
    "船": "ship",
    "船只": "ship",
    "舰船": "ship",
    "体育场": "stadium",
    "储油罐": "storagetank",
    "油罐": "storagetank",
    "网球场": "tenniscourt",
    "火车站": "trainstation",
    "车辆": "vehicle",
    "汽车": "vehicle",
    "车": "vehicle",
    "风车": "windmill",
    "风力发电机": "windmill",
}


def normalize_class_name(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if not normalized:
        return None
    for class_name in DIOR_CLASSES:
        if normalized == class_name.lower().replace("-", ""):
            return class_name
    alias_value = CLASS_ALIASES.get(value.strip()) or CLASS_ALIASES.get(normalized)
    return alias_value


def import_file(path: Path, module_name: str) -> None:
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def rbox_to_polygon(box: list[float]) -> list[tuple[float, float]]:
    cx, cy, w, h, angle = box
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


def label_color(label: int) -> tuple[int, int, int]:
    palette = [
        (220, 20, 60),
        (0, 128, 255),
        (40, 180, 99),
        (255, 140, 0),
        (155, 89, 182),
        (46, 204, 113),
        (241, 196, 15),
        (231, 76, 60),
        (52, 152, 219),
        (127, 140, 141),
    ]
    return palette[label % len(palette)]


def draw_detections(image_path: str, detections: list[dict], task: str, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        color = label_color(int(det["class_id"]))
        text = f'{det["class_name"]} {det["score"]:.2f}'
        if task == "horizontal":
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            text_pos = (x1, max(0, y1 - 14))
        else:
            poly = [tuple(p) for p in det["polygon"]]
            draw.line(poly + [poly[0]], fill=color, width=2)
            text_pos = poly[0]
        draw.text(text_pos, text, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def prepare_inference_image(image_path: str, output_dir: Path, size: int = 800) -> tuple[str, str | None]:
    image = Image.open(image_path).convert("RGB")
    if image.size == (size, size):
        return image_path, None

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    resized = image.copy()
    resized.thumbnail((size, size), Image.Resampling.BILINEAR)
    x = (size - resized.width) // 2
    y = (size - resized.height) // 2
    canvas.paste(resized, (x, y))

    prepared_dir = output_dir / "_prepared_inputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / f"{Path(image_path).stem}_letterbox_{size}.jpg"
    canvas.save(prepared_path)
    return str(prepared_path), str(prepared_path)


def detections_from_sample(sample, score_thr: float, task: str) -> list[dict]:
    pred = sample.pred_instances
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        score = float(score)
        if score < score_thr:
            continue
        label_id = int(label)
        item = {
            "class_id": label_id,
            "class_name": DIOR_CLASSES[label_id] if label_id < len(DIOR_CLASSES) else f"class_{label_id}",
            "score": score,
        }
        if task == "horizontal":
            item["bbox"] = [float(x) for x in bbox.tolist()]
        else:
            rbox = [float(x) for x in bbox.tolist()]
            item["rbox"] = rbox
            item["polygon"] = [[float(x), float(y)] for x, y in rbox_to_polygon(rbox)]
        detections.append(item)
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections


def load_horizontal_model(device: str):
    from mmengine.config import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(str(HORIZONTAL_CONFIG))
    cfg.model.backbone.pretrained = None
    cfg.test_pipeline = [step for step in cfg.test_pipeline if step.get("type") != "LoadAnnotations"]
    cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
    return init_detector(cfg, str(HORIZONTAL_CKPT), device=device)


def load_rotated_model(device: str):
    import mmdet

    # mmrotate 1.0.0rc1 only checks this string during import.
    mmdet.__version__ = "3.0.0"
    import mmrotate  # noqa: F401
    from mmengine.config import Config
    from mmdet.apis import init_detector

    import_file(ROTATED_ROOT / "mmrotate" / "models" / "backbones" / "vit_rvsa_mtp_branches.py", "mtp_rotated_rvsa")

    cfg = Config.fromfile(str(ROTATED_CONFIG))
    cfg.model.backbone.pretrained = None
    cfg.test_pipeline = [
        step for step in cfg.test_pipeline if step.get("type") not in {"mmdet.LoadAnnotations", "ConvertBoxType"}
    ]
    cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
    return init_detector(cfg, str(ROTATED_CKPT), device=device)


def run_one(task: str, req: dict) -> dict:
    from mmdet.apis import inference_detector

    image_path = req["image_path"]
    score_thr = float(req.get("score_thr") or 0.3)
    target_class = normalize_class_name(req.get("target_class"))
    device = req.get("device") or "cuda:0"
    output_dir = Path(req["output_dir"]) / task
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_image_path, prepared_image_path = prepare_inference_image(image_path, output_dir)

    if task == "horizontal":
        model = load_horizontal_model(device)
        config = HORIZONTAL_CONFIG
        checkpoint = HORIZONTAL_CKPT
    elif task == "rotated":
        model = load_rotated_model(device)
        config = ROTATED_CONFIG
        checkpoint = ROTATED_CKPT
    else:
        raise ValueError("MTP task must be 'horizontal', 'rotated', or 'both'")

    sample = inference_detector(model, inference_image_path)
    detections = detections_from_sample(sample, score_thr, task)
    if target_class:
        detections = [det for det in detections if det.get("class_name") == target_class]
    stem = Path(image_path).stem
    vis_path = output_dir / f"{stem}_mtp_{task}.jpg"
    json_path = output_dir / f"{stem}_mtp_{task}.json"
    draw_detections(inference_image_path, detections, task, vis_path)

    payload = {
        "model": "MTP",
        "task": task,
        "image": image_path,
        "inference_image": inference_image_path,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "score_thr": score_thr,
        "target_class": target_class,
        "num_detections": len(detections),
        "detections": detections,
        "visualization": str(vis_path),
        "raw_result": str(json_path),
    }
    if prepared_image_path:
        payload["resized_for_inference"] = prepared_image_path
        payload["note"] = "Input was letterboxed to 800x800 because this MTP checkpoint expects fixed 800x800 inputs."
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())
    task = req.get("task") or "rotated"
    if task == "both":
        result = {
            "model": "MTP",
            "task": "both",
            "image": req["image_path"],
            "horizontal": run_one("horizontal", req),
            "rotated": run_one("rotated", req),
        }
    else:
        aliases = {
            "hbb": "horizontal",
            "horizontal_detect": "horizontal",
            "obb": "rotated",
            "rotated_detect": "rotated",
        }
        result = run_one(aliases.get(task, task), req)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
