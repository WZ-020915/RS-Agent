from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model_wrappers import run_remote_sensing_agent_system  # noqa: E402
from model_wrappers.agent_system import (  # noqa: E402
    best_image_to_text_match,
    format_class_distribution,
    synthesize_answer_without_llm,
)
from model_wrappers.test_agent_chain import summarize_result  # noqa: E402


DATASET_ROOT = REPO_ROOT / "model_wrappers" / "dataset_test"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "model_wrappers" / "dataset_test_reports"
FONT_CACHE_DIR = REPO_ROOT / "model_wrappers" / "assets" / "fonts"
DEFAULT_CJK_FONT = FONT_CACHE_DIR / "NotoSansCJKsc-Regular.otf"
RAW_AIR_POLARSAR_SEG_ROOT = Path("/root/autodl-tmp/SARMAE-main/Raw_AIR-PolarSAR-Seg")

SKYEYE_QUERY = "请描述这张遥感图像。"
SAR_DETECT_QUERY = "这是一张 SAR 图像，图中有几个船？"
SAR_SEG_QUERY = "请对这张SAR图像做语义分割。"
DOFA_SEG_QUERY = "请对这张遥感图像做语义分割。"
SATTXT_QUERY = "从候选文本中找出最匹配这张图的描述。"
DEHAZE_QUERY = "请对这张云雾遥感图像去云/去雾。"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

SKYEYE_DEMO_CAPTION_QUESTIONS = [
    "briefly describe the image",
    "explain what you see in the image in details",
    "Summarize this image in a 10 words",
]

SKYEYE_DEMO_VQA_QUESTIONS = [
    {
        "suffix": "moving_ship",
        "image_id": "05864_0000.png",
        "query": "Is there a moving ship in this picture?",
        "gt_text": "Demo VQA GT: No moving ship is annotated in this sampled bridge image.",
    },
    {
        "suffix": "less_ships_than_cars",
        "image_id": "05870_0000.png",
        "query": "Are there less ships than cars in this picture?",
        "gt_text": "Demo VQA GT: Yes. The sampled image has cars in the caption/VQA annotations and no annotated ships.",
    },
]

SKYEYE_DEMO_REFERRING_QUESTIONS = [
    {
        "choice": "gray_overpass_right",
        "source_question_id": 8788,
        "query": "where can I locate the gray overpass on the right?",
    },
    {
        "choice": "red_car_next_to_grey_car",
        "source_question_id": 8786,
        "query": "where is the red car next to a grey car?",
    },
]

SATTXT_DOTA_TEXTS = [
    "vehicles on roads or parking lots",
    "harbor with ships",
    "ships on water",
    "sports fields",
    "swimming pools",
    "road junction or roundabout",
    "airport or airplanes",
    "bridges or overpasses",
    "dense urban buildings",
    "open land",
]

DOTA_TARGET_LABELS = {
    "airplane": {"plane"},
    "baseballfield": {"baseball-diamond"},
    "bridge": {"bridge"},
    "groundtrackfield": {"ground-track-field"},
    "harbor": {"harbor"},
    "ship": {"ship"},
    "vehicle": {"small-vehicle", "large-vehicle"},
}

MTP_DOTA_DEFAULT_QUESTIONS = [
    {
        "suffix": "vehicle_presence",
        "query": "图中是否有车？",
        "target_class": "vehicle",
        "gt_name": "车辆",
        "mode": "presence",
    },
    {
        "suffix": "vehicle_count",
        "query": "图中大概有多少辆车？",
        "target_class": "vehicle",
        "gt_name": "车辆",
        "mode": "count",
    },
    {
        "suffix": "ship_presence",
        "query": "图中是否能看到船只？",
        "target_class": "ship",
        "gt_name": "船只",
        "mode": "presence",
    },
    {
        "suffix": "vehicle_abundance",
        "query": "图中的车辆是少量还是较多？",
        "target_class": "vehicle",
        "gt_name": "车辆",
        "mode": "abundance",
        "many_threshold": 20,
    },
    {
        "suffix": "ship_count",
        "query": "图中大概有多少艘船？",
        "target_class": "ship",
        "gt_name": "船只",
        "mode": "count",
    },
    {
        "suffix": "vehicle_ship_compare",
        "query": "图中车辆和船只哪一类更多？",
        "target_classes": ["vehicle", "ship"],
        "gt_names": ["车辆", "船只"],
        "mode": "compare",
    },
    {
        "suffix": "harbor_presence",
        "query": "图中是否有港口区域？",
        "target_class": "harbor",
        "gt_name": "港口",
        "mode": "presence",
    },
    {
        "suffix": "airplane_presence",
        "query": "图中是否有飞机？",
        "target_class": "airplane",
        "gt_name": "飞机",
        "mode": "presence",
    },
    {
        "suffix": "bridge_presence",
        "query": "图中能否看到桥梁？",
        "target_class": "bridge",
        "gt_name": "桥梁",
        "mode": "presence",
    },
    {
        "suffix": "baseballfield_presence",
        "query": "图中有没有棒球场？",
        "target_class": "baseballfield",
        "gt_name": "棒球场",
        "mode": "presence",
    },
]

MTP_DOTA_OBJECT_QUESTIONS = [
    *MTP_DOTA_DEFAULT_QUESTIONS,
    {
        "suffix": "harbor_count",
        "query": "图中有几个港口？",
        "target_class": "harbor",
        "gt_name": "港口",
        "mode": "count",
    },
    {
        "suffix": "groundtrackfield_presence",
        "query": "图中是否有田径场？",
        "target_class": "groundtrackfield",
        "gt_name": "田径场",
        "mode": "presence",
    },
]


@dataclass
class Case:
    case_id: str
    task: str
    query: str
    image_path: Path | None = None
    gt_text: str = ""
    gt_image_path: Path | None = None
    gt_mask_path: Path | None = None
    extra_gt_image_path: Path | None = None
    dataset: str | None = None
    sample_id: str | None = None
    input_path: Path | None = None
    texts: list[str] | None = None
    planner_json: dict[str, Any] | None = None
    reference_only: bool = False


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_to_root(value: str | Path | None, root: Path = DATASET_ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_manifest() -> dict[str, Any]:
    return load_json(DATASET_ROOT / "manifest.json")


def skyeyegpt_records(annotation_name: str) -> list[dict[str, Any]]:
    root = DATASET_ROOT / "skyeyegpt"
    records = load_json(root / annotation_name)
    return [item for item in records if (root / "images" / str(item.get("image_id", ""))).exists()]


def skyeyegpt_image_paths() -> list[Path]:
    image_dir = DATASET_ROOT / "skyeyegpt" / "images"
    return sorted(path for path in image_dir.iterdir() if path.is_file())


def group_records_by_image(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("image_id", "")), []).append(record)
    return grouped


def skyeyegpt_referring_record_by_question_id(question_id: int) -> dict[str, Any] | None:
    for record in skyeyegpt_records("VRSBench_EVAL_referring.json"):
        if int(record.get("question_id", -1)) == question_id:
            return record
    return None


def skyeyegpt_referring_gt_visual(record: dict[str, Any]) -> Path | None:
    corners = record.get("obj_corner")
    image_id = str(record.get("image_id") or "")
    if not image_id or not isinstance(corners, list) or len(corners) < 8:
        return None
    image_path = DATASET_ROOT / "skyeyegpt" / "images" / image_id
    if not image_path.exists():
        return None
    out_dir = DATASET_ROOT / "skyeyegpt" / "gt_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(image_id).stem}_q{record.get('question_id', 'ref')}_gt.png"
    if out_path.exists():
        return out_path

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    points = []
    for idx in range(0, 8, 2):
        x = max(0.0, min(1.0, float(corners[idx]))) * width
        y = max(0.0, min(1.0, float(corners[idx + 1]))) * height
        points.append((x, y))
    draw = ImageDraw.Draw(image)
    draw.line(points + [points[0]], fill=(255, 0, 0), width=max(3, width // 120))
    label = str(record.get("obj_cls") or "GT")
    draw.rectangle((8, 8, 18 + len(label) * 9, 32), fill=(255, 0, 0))
    draw.text((12, 12), label, fill=(255, 255, 255))
    image.save(out_path)
    return out_path


def sar_detect_counts() -> tuple[dict[str, int], str]:
    annotation_dir = DATASET_ROOT / "sarmae_detect" / "annotations"
    annotation_path = annotation_dir / "train_test2017.json"
    if not annotation_path.exists():
        annotation_path = annotation_dir / "test2017.json"
    coco = load_json(annotation_path)
    id_to_name = {int(item["id"]): item["file_name"] for item in coco.get("images", [])}
    counts = {name: 0 for name in id_to_name.values()}
    for ann in coco.get("annotations", []):
        name = id_to_name.get(int(ann.get("image_id", -1)))
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return counts, annotation_path.name


def class_count_text(counts: dict[str, Any] | None) -> str:
    if not counts:
        return ""
    return ", ".join(f"class {key}: {value}" for key, value in sorted(counts.items(), key=lambda kv: int(kv[0])))


def mask_class_counts(mask_path: Path | None) -> dict[str, int]:
    if not mask_path or not mask_path.exists():
        return {}
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        flat = mask.reshape(-1, mask.shape[-1])
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        return {"/".join(str(int(v)) for v in color): int(count) for color, count in zip(colors, counts)}
    classes, counts = np.unique(mask, return_counts=True)
    return {str(int(class_id)): int(count) for class_id, count in zip(classes, counts)}


def raw_air_polarsar_records() -> list[dict[str, Any]]:
    manifest_path = RAW_AIR_POLARSAR_SEG_ROOT / "test" / "manifest.json"
    if not manifest_path.exists():
        return []
    records = load_json(manifest_path)
    if not isinstance(records, list):
        return []
    return [record for record in records if Path(str(record.get("input", ""))).exists()]


def dehazeformer_image_paths() -> list[Path]:
    roots = [DATASET_ROOT / "dehazeformer", DATASET_ROOT / "RICE"]
    for root in roots:
        if not root.exists():
            continue
        images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
        if images:
            return images
    return []


def dota_target_count(label_path: Path | None, target_class: str) -> int | None:
    if not label_path or not label_path.exists():
        return None
    labels = DOTA_TARGET_LABELS.get(target_class, {target_class})
    count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[8] in labels:
            count += 1
    return count


def dota_label_counts(label_path: Path | None) -> dict[str, int]:
    if not label_path or not label_path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 9:
            label = parts[8]
            counts[label] = counts.get(label, 0) + 1
    return counts


def sattxt_dota_gt_text(label_path: Path | None) -> str:
    counts = dota_label_counts(label_path)
    if not counts:
        return "GT DOTA objects: none or label unavailable; likely open land/background scene"
    phrases = []
    vehicle_count = counts.get("small-vehicle", 0) + counts.get("large-vehicle", 0)
    if vehicle_count:
        phrases.append(f"vehicles: {vehicle_count}")
    for label, name in [
        ("harbor", "harbors"),
        ("ship", "ships"),
        ("baseball-diamond", "baseball diamonds"),
        ("ground-track-field", "ground track fields"),
        ("soccer-ball-field", "soccer fields"),
        ("swimming-pool", "swimming pools"),
        ("roundabout", "roundabouts"),
        ("plane", "airplanes"),
        ("bridge", "bridges"),
        ("overpass", "overpasses"),
    ]:
        if counts.get(label):
            phrases.append(f"{name}: {counts[label]}")
    if not phrases:
        phrases = [f"{label}: {count}" for label, count in sorted(counts.items())]
    return "GT DOTA objects: " + ", ".join(phrases)


def dota_vehicle_count(label_path: Path | None) -> int | None:
    return dota_target_count(label_path, "vehicle")


def dota_target_gt_visual(image_path: Path | None, label_path: Path | None, target_class: str) -> Path | None:
    if not image_path or not image_path.exists() or not label_path or not label_path.exists():
        return None
    out_dir = DATASET_ROOT / "mtp_dota" / "gt_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_{target_class}_gt.png"
    if out_path.exists():
        return out_path

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(4, min(image.size) // 180)
    labels = DOTA_TARGET_LABELS.get(target_class, {target_class})
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 9 or parts[8] not in labels:
            continue
        coords = [float(value) for value in parts[:8]]
        points = [(coords[idx], coords[idx + 1]) for idx in range(0, 8, 2)]
        draw.line(points + [points[0]], fill=(255, 0, 0), width=width)
    image.save(out_path)
    return out_path


def dota_vehicle_gt_visual(image_path: Path | None, label_path: Path | None) -> Path | None:
    return dota_target_gt_visual(image_path, label_path, "vehicle")


def make_planner(agent: str, task: str, task_type: str) -> dict[str, Any]:
    params = {} if agent == "dofa" else {"task": task}
    return {
        "task_type": task_type,
        "reasoning": f"Fixed planner for dataset_test batch task {task_type}.",
        "nodes": [
            {
                "id": "node_1",
                "agent": agent,
                "params": params,
                "depends_on": [],
            }
        ],
    }


def mtp_dota_case_from_spec(rec: dict[str, Any], spec: dict[str, Any]) -> Case:
    image = rel_to_root(rec["image"])
    label = rel_to_root(rec.get("label"))
    mode = str(spec["mode"])

    if mode == "compare":
        target_classes = [str(value) for value in spec["target_classes"]]
        gt_names = [str(value) for value in spec["gt_names"]]
        counts = [dota_target_count(label, target_class) for target_class in target_classes]
        gt_visual = dota_target_gt_visual(image, label, target_classes[0])
        if any(count is None for count in counts):
            gt_text = f"GT {'/'.join(gt_names)}: N/A（未找到 DOTA label）"
        else:
            count_pairs = list(zip(gt_names, counts))
            max_count = max(int(count) for count in counts if count is not None)
            detail = "，".join(f"{name}: {count}" for name, count in count_pairs)
            if sum(1 for _, count in count_pairs if count == max_count) > 1:
                gt_text = f"GT 比较: 数量相同（{detail}，来自 DOTA label）"
            else:
                winner = next(name for name, count in count_pairs if count == max_count)
                gt_text = f"GT 比较: {winner}更多（{detail}，来自 DOTA label）"
        nodes = [
            {
                "id": f"node_{idx + 1}",
                "agent": "mtp",
                "params": {"task": "horizontal", "target_class": target_class},
                "depends_on": [],
            }
            for idx, target_class in enumerate(target_classes)
        ]
        planner_json = {
            "task_type": "detection_compare",
            "reasoning": "Fixed planner for comparing MTP detection counts across target classes.",
            "nodes": nodes,
        }
    else:
        target_class = str(spec["target_class"])
        count = dota_target_count(label, target_class)
        gt_visual = dota_target_gt_visual(image, label, target_class)
        gt_name = str(spec["gt_name"])
        if count is None:
            gt_text = f"GT {gt_name}: N/A（未找到 DOTA label）"
        elif mode == "presence":
            gt_text = f"GT 是否有{gt_name}: {'是' if count > 0 else '否'}（DOTA label 数量: {count}）"
        elif mode == "abundance":
            threshold = int(spec.get("many_threshold", 20))
            gt_text = f"GT {gt_name}数量级: {'较多' if count >= threshold else '少量'}（DOTA label 数量: {count}，阈值: {threshold}）"
        elif mode == "detect":
            gt_text = f"GT {gt_name}检测目标数: {count}（来自 DOTA label）"
        else:
            gt_text = f"GT {gt_name}数量: {count}（来自 DOTA label）"

        task_type = {
            "presence": "detection_vqa",
            "abundance": "detection_abundance",
            "detect": "detection_request",
        }.get(mode, "counting")
        planner_json = {
            "task_type": task_type,
            "reasoning": f"Fixed planner for optical {target_class} detection with MTP.",
            "nodes": [
                {
                    "id": "node_1",
                    "agent": "mtp",
                    "params": {"task": "horizontal", "target_class": target_class},
                    **({"answer_options": {"many_threshold": int(spec.get("many_threshold", 20))}} if mode == "abundance" else {}),
                    "depends_on": [],
                }
            ],
        }

    return Case(
        case_id=f"mtp_dota_{spec['suffix']}_{rec['sample_id']}",
        task=f"mtp_dota_object_{mode}",
        query=str(spec["query"]),
        image_path=image,
        gt_text=gt_text,
        gt_image_path=gt_visual,
        planner_json=planner_json,
    )


def build_cases(tasks: set[str], max_cases: int, skyeyegpt_refer_choice: str = "all") -> list[Case]:
    manifest = load_manifest()
    cases: list[Case] = []

    def take(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return records if max_cases == 0 else records[:max_cases]

    if "skyeyegpt" in tasks:
        image_paths = skyeyegpt_image_paths()
        image_paths = image_paths if max_cases == 0 else image_paths[:max_cases]
        cap_by_image = group_records_by_image(skyeyegpt_records("VRSBench_EVAL_Cap.json"))
        vqa_by_image = group_records_by_image(skyeyegpt_records("VRSBench_EVAL_vqa.json"))

        for image in image_paths:
            image_cases: list[Case] = []
            cap_rec = (cap_by_image.get(image.name) or [{}])[0]
            for prompt_idx, query in enumerate(SKYEYE_DEMO_CAPTION_QUESTIONS, start=1):
                image_cases.append(
                    Case(
                        case_id=f"skyeyegpt_caption_p{prompt_idx}_{image.stem}",
                        task="skyeyegpt_caption",
                        query=query,
                        image_path=image,
                        gt_text=f"GT caption: {cap_rec.get('ground_truth', 'N/A')}",
                        planner_json=make_planner("skyeyegpt", "caption", "caption"),
                    )
                )

            for spec in SKYEYE_DEMO_VQA_QUESTIONS:
                if str(spec["image_id"]) != image.name:
                    continue
                image_cases.append(
                    Case(
                        case_id=f"skyeyegpt_demo_vqa_{spec['suffix']}_{image.stem}",
                        task="skyeyegpt_vqa",
                        query=str(spec["query"]),
                        image_path=image,
                        gt_text=str(spec["gt_text"]),
                        planner_json=make_planner("skyeyegpt", "vqa", "vqa"),
                    )
                )

            for spec in SKYEYE_DEMO_REFERRING_QUESTIONS:
                if skyeyegpt_refer_choice != "all" and skyeyegpt_refer_choice != spec["choice"]:
                    continue
                rec = skyeyegpt_referring_record_by_question_id(int(spec["source_question_id"]))
                if rec is None or str(rec.get("image_id")) != image.name:
                    continue
                image_cases.append(
                    Case(
                        case_id=f"skyeyegpt_demo_referring_{spec['choice']}_{image.stem}",
                        task="skyeyegpt_referring",
                        query=str(spec["query"]),
                        image_path=image,
                        gt_text=(
                            f"GT referring: {rec.get('ground_truth', 'N/A')}; "
                            f"obj_cls={rec.get('obj_cls', 'N/A')}; "
                            f"source_expression={rec.get('question', 'N/A')}; demo_choice={spec['choice']}"
                        ),
                        gt_image_path=skyeyegpt_referring_gt_visual(rec),
                        planner_json=make_planner("skyeyegpt", "grounding", "grounding"),
                    )
                )

            for rec in vqa_by_image.get(image.name, []):
                image_cases.append(
                    Case(
                        case_id=f"skyeyegpt_vqa_q{rec['question_id']}_{image.stem}",
                        task="skyeyegpt_vqa",
                        query=str(rec.get("question") or "请回答这张遥感图像相关的问题。"),
                        image_path=image,
                        gt_text=f"GT answer: {rec.get('ground_truth', 'N/A')}; type: {rec.get('type', 'N/A')}",
                        planner_json=make_planner("skyeyegpt", "vqa", "vqa"),
                    )
                )

            cases.extend(image_cases[:5])

    if "dehazeformer" in tasks:
        for image in dehazeformer_image_paths() if max_cases == 0 else dehazeformer_image_paths()[:max_cases]:
            cases.append(
                Case(
                    case_id=f"dehazeformer_{image.stem}",
                    task="dehazeformer_dehaze",
                    query=DEHAZE_QUERY,
                    image_path=image,
                    gt_text="DehazeFormer 去云/去雾任务：输入为云雾/雾霾遥感图像，输出为复原图像。",
                    planner_json={
                        "task_type": "dehazing",
                        "reasoning": "Fixed planner for single-image dehazing with DehazeFormer.",
                        "nodes": [
                            {
                                "id": "node_1",
                                "agent": "dehazeformer",
                                "params": {"model_name": "dehazeformer-b", "exp": "reside6k"},
                                "depends_on": [],
                            }
                        ],
                    },
                )
            )

    if "sarmae_detect" in tasks:
        counts, annotation_name = sar_detect_counts()
        for rec in take(manifest["sarmae_detect"]["records"]):
            image = rel_to_root(rec["image"])
            count = counts.get(image.name if image else "")
            gt_text = (
                f"GT 船只数量: {count}（来自 {annotation_name}）"
                if count is not None
                else f"GT 船只数量: N/A（{annotation_name} 未找到该图）"
            )
            cases.append(
                Case(
                    case_id=f"sarmae_detect_{rec['sample_id']}",
                    task="sarmae_detect",
                    query=SAR_DETECT_QUERY,
                    image_path=image,
                    gt_text=gt_text,
                    planner_json=make_planner("sarmae", "detect", "sar_detection_count"),
                )
            )

    if "mtp_dota" in tasks:
        for idx, rec in enumerate(take(manifest["mtp_dota"]["records"])):
            spec = MTP_DOTA_DEFAULT_QUESTIONS[idx % len(MTP_DOTA_DEFAULT_QUESTIONS)]
            cases.append(mtp_dota_case_from_spec(rec, spec))

    if "mtp_dota_object_vqa" in tasks:
        for rec in take(manifest["mtp_dota"]["records"]):
            for spec in MTP_DOTA_OBJECT_QUESTIONS:
                cases.append(mtp_dota_case_from_spec(rec, spec))

    if "sarmae_segment" in tasks:
        for rec in take(manifest["sarmae_segment"]["records"]):
            cases.append(
                Case(
                    case_id=f"sarmae_segment_{rec['sample_id']}",
                    task="sarmae_segment",
                    query=SAR_SEG_QUERY,
                    image_path=rel_to_root(rec["image"]),
                    gt_text=f"GT 类别像素数: {class_count_text(rec.get('class_pixel_counts'))}",
                    gt_image_path=rel_to_root(rec.get("mask_color")),
                    gt_mask_path=rel_to_root(rec.get("mask")),
                    extra_gt_image_path=rel_to_root(rec.get("overlay")),
                    planner_json=make_planner("sarmae", "segment", "sar_segmentation"),
                )
            )

    if "sarmae_segment_raw_air" in tasks:
        for rec in take(raw_air_polarsar_records()):
            sample_id = str(rec["sample_id"])
            annotation = Path(str(rec.get("annotation", "")))
            label = Path(str(rec.get("label", "")))
            overlay = Path(str(rec.get("overlay", "")))
            gt_counts = class_count_text(mask_class_counts(annotation))
            metric_text = ""
            if rec.get("pixel_accuracy") is not None and rec.get("miou") is not None:
                metric_text = f"; dataset precomputed pixel_acc={float(rec['pixel_accuracy']):.3f}, mIoU={float(rec['miou']):.3f}"
            cases.append(
                Case(
                    case_id=f"sarmae_segment_raw_air_{sample_id}",
                    task="sarmae_segment_raw_air",
                    query=SAR_SEG_QUERY,
                    image_path=Path(str(rec["input"])),
                    gt_text=(
                        "GT 数据集: Raw_AIR-PolarSAR-Seg test，极化通道 HH；"
                        f"GT 类别像素数: {gt_counts or 'N/A'}{metric_text}"
                    ),
                    gt_image_path=label if label.exists() else None,
                    gt_mask_path=annotation if annotation.exists() else None,
                    extra_gt_image_path=overlay if overlay.exists() else None,
                    planner_json=make_planner("sarmae", "segment", "sar_segmentation"),
                )
            )

    if "dofa_hdf5" in tasks:
        hdf5_root = DATASET_ROOT / "dofa_hdf5" / "m-pv4ger-seg"
        hdf5_paths = sorted((hdf5_root / "hdf5").glob("*.hdf5"))
        for path in hdf5_paths if max_cases == 0 else hdf5_paths[:max_cases]:
            sample_id = path.stem
            cases.append(
                Case(
                    case_id=f"dofa_hdf5_{sample_id}",
                    task="dofa_seg_hdf5",
                    query=DOFA_SEG_QUERY,
                    image_path=hdf5_root / "visuals" / f"{sample_id}_rgb.png",
                    gt_text=(
                        "GT: m-pv4ger-seg 二分类语义分割；类别为 background 和 "
                        "photovoltaic_panel（光伏板），GT 高亮区域表示光伏板。"
                    ),
                    gt_image_path=hdf5_root / "visuals" / f"{sample_id}_label_color.png",
                    gt_mask_path=hdf5_root / "masks" / f"{sample_id}_label.png",
                    extra_gt_image_path=hdf5_root / "visuals" / f"{sample_id}_overlay.png",
                    dataset="m-pv4ger-seg",
                    input_path=path,
                    planner_json=make_planner("dofa", "segmentation", "semantic_segmentation"),
                )
            )

    if "dofa_png_val" in tasks:
        dofa_png_root = DATASET_ROOT / "dofa_png"
        png_records = []
        for image_path in sorted(
            path
            for path in dofa_png_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTS
            and ".ipynb_checkpoints" not in path.parts
        ):
            sample_id = image_path.stem.removesuffix("_rgb")
            png_records.append(
                {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "mask": None,
                    "mask_color": None,
                    "overlay": None,
                    "class_pixel_counts": None,
                }
            )
        png_records = [
            *png_records,
            *[
                rec for rec in manifest.get("dofa_png_val", {}).get("records", [])
                if (image := rel_to_root(rec.get("image"))) is not None and image.exists()
            ],
        ]
        if not png_records:
            hdf5_root = DATASET_ROOT / "dofa_hdf5" / "m-pv4ger-seg"
            fallback_records = []
            for rgb_path in sorted((hdf5_root / "visuals").glob("*_rgb.png")):
                sample_id = rgb_path.name.removesuffix("_rgb.png")
                fallback_records.append(
                    {
                        "sample_id": sample_id,
                        "image": str(rgb_path),
                        "mask": str(hdf5_root / "masks" / f"{sample_id}_label.png"),
                        "mask_color": str(hdf5_root / "visuals" / f"{sample_id}_label_color.png"),
                        "overlay": str(hdf5_root / "visuals" / f"{sample_id}_overlay.png"),
                        "class_pixel_counts": None,
                    }
                )
            png_records = fallback_records
        for rec in take(png_records):
            cases.append(
                Case(
                    case_id=f"dofa_png_val_{rec['sample_id']}",
                    task="dofa_png_segmentation",
                    query=DOFA_SEG_QUERY,
                    image_path=rel_to_root(rec["image"]),
                    gt_text=f"PNG 参考集 GT 类别像素数: {class_count_text(rec.get('class_pixel_counts'))}",
                    gt_image_path=rel_to_root(rec.get("mask_color")),
                    gt_mask_path=rel_to_root(rec.get("mask")),
                    extra_gt_image_path=rel_to_root(rec.get("overlay")),
                    dataset="m-pv4ger-seg",
                    input_path=rel_to_root(rec["image"]),
                    planner_json=make_planner("dofa", "segmentation", "semantic_segmentation"),
                )
            )

    if "sattxt" in tasks:
        texts = list(SATTXT_DOTA_TEXTS)
        candidate_text = "候选文本: " + "; ".join(texts)
        for rec in take(manifest["mtp_dota"]["records"]):
            image = rel_to_root(rec["image"])
            label = rel_to_root(rec.get("label"))
            cases.append(
                Case(
                    case_id=f"sattxt_mtp_dota_{rec['sample_id']}",
                    task="sattxt_retrieval",
                    query=f"{SATTXT_QUERY} {candidate_text}",
                    image_path=image,
                    gt_text=sattxt_dota_gt_text(label),
                    texts=texts,
                    planner_json=make_planner("sattxt", "retrieval", "image_text_retrieval"),
                )
            )

    return cases


def extract_json_from_stdout(stdout: str) -> dict[str, Any]:
    stdout = stdout.strip()
    if not stdout:
        raise ValueError("empty stdout")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stdout[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object from test_agent_chain")
    return value


def synthesize_dataset_answer_without_llm(
    context: dict[str, Any],
    plan: dict[str, Any],
    agent_results: dict[str, Any],
) -> str:
    def mtp_node_result(node: dict[str, Any]) -> tuple[dict[str, Any], str, int] | None:
        result = agent_results.get(str(node.get("id")))
        if not isinstance(result, dict) or result.get("model") != "MTP":
            return None
        if result.get("status") == "error":
            return result, "", 0
        target = str(result.get("target_class") or (node.get("params") or {}).get("target_class") or "目标")
        return result, target, int(result.get("num_detections", 0))

    if plan.get("task_type") == "detection_vqa":
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            item = mtp_node_result(node)
            if item is None:
                continue
            result, target, count = item
            if result.get("status") == "error":
                return f"MTP 调用失败，错误是 {result.get('error_type')}: {result.get('error')}"
            return f"{'是' if count > 0 else '否'}，检测到 {count} 个 {target} 类目标。"
    if plan.get("task_type") == "detection_abundance":
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            item = mtp_node_result(node)
            if item is None:
                continue
            result, target, count = item
            if result.get("status") == "error":
                return f"MTP 调用失败，错误是 {result.get('error_type')}: {result.get('error')}"
            threshold = int((node.get("answer_options") or {}).get("many_threshold", 20))
            return f"{'较多' if count >= threshold else '少量'}，检测到 {count} 个 {target} 类目标。"
    if plan.get("task_type") == "detection_request":
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            item = mtp_node_result(node)
            if item is None:
                continue
            result, target, count = item
            if result.get("status") == "error":
                return f"MTP 调用失败，错误是 {result.get('error_type')}: {result.get('error')}"
            return f"已检测并框出 {target} 类目标，共 {count} 个。结果图: {result.get('visualization')}"
    if plan.get("task_type") == "detection_compare":
        counts = []
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            item = mtp_node_result(node)
            if item is None:
                continue
            result, target, count = item
            if result.get("status") == "error":
                return f"MTP 调用失败，错误是 {result.get('error_type')}: {result.get('error')}"
            counts.append((target, count))
        if counts:
            max_count = max(count for _, count in counts)
            winners = [target for target, count in counts if count == max_count]
            detail = "，".join(f"{target}: {count}" for target, count in counts)
            if len(winners) > 1:
                return f"数量相同。检测数量为：{detail}。"
            return f"{'、'.join(winners)}更多。检测数量为：{detail}。"
    return synthesize_answer_without_llm(context, plan, agent_results)


def run_case_direct(case: Case, output_dir: Path, use_fixed_planner: bool = False) -> dict[str, Any]:
    if case.reference_only:
        return {
            "status": "reference_only",
            "query": case.query,
            "answer": "该 case 仅整理 PNG 参考集 GT；现有 DOFA wrapper 需要 HDF5 输入，未对 PNG 直接推理。",
            "agent_results": {},
        }
    result = run_remote_sensing_agent_system(
        case.query,
        image_path=str(case.image_path) if case.image_path else None,
        image_paths=[str(case.image_path)] if case.image_path else None,
        texts=case.texts,
        dataset=case.dataset,
        sample_id=case.sample_id,
        input_path=str(case.input_path) if case.input_path else None,
        output_dir=str(output_dir / "agent_outputs" / case.case_id),
        planner_json=case.planner_json if use_fixed_planner else None,
        execute=True,
        synthesizer=synthesize_dataset_answer_without_llm,
    )
    return result


def run_case_cli(case: Case) -> dict[str, Any]:
    if case.reference_only:
        return {
            "status": "reference_only",
            "query": case.query,
            "answer": "该 case 仅整理 PNG 参考集 GT；现有 DOFA wrapper 需要 HDF5 输入，未对 PNG 直接推理。",
        }
    cmd = [
        sys.executable,
        str(REPO_ROOT / "model_wrappers" / "test_agent_chain.py"),
        "--query",
        case.query,
        "--execute",
        "--no-llm-final",
    ]
    if case.image_path:
        cmd += ["--image-path", str(case.image_path)]
    if case.texts:
        cmd += ["--texts", json.dumps(case.texts, ensure_ascii=False)]
    if case.dataset:
        cmd += ["--dataset", case.dataset]
    if case.sample_id:
        cmd += ["--sample-id", case.sample_id]
    if case.input_path:
        cmd += ["--input-path", str(case.input_path)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {
            "status": "error",
            "answer": f"test_agent_chain.py failed with return code {proc.returncode}",
            "stderr": proc.stderr[-4000:],
            "stdout": proc.stdout[-4000:],
            "command": cmd,
        }
    try:
        return extract_json_from_stdout(proc.stdout)
    except Exception as exc:
        return {
            "status": "error",
            "answer": f"Failed to parse stdout JSON: {type(exc).__name__}: {exc}",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "command": cmd,
        }


def first_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    agent_results = result.get("agent_results") or {}
    if isinstance(agent_results, dict):
        for value in agent_results.values():
            if isinstance(value, dict):
                return value
    summary = result.get("agent_results") or {}
    if isinstance(summary, dict):
        for value in summary.values():
            if isinstance(value, dict):
                return value
    return {}


def answer_from_result(result: dict[str, Any]) -> str:
    answer = result.get("answer")
    if answer:
        return str(answer)
    agent = first_agent_result(result)
    if agent.get("status") == "error":
        return f"ERROR {agent.get('error_type')}: {agent.get('error')}"
    if agent.get("model") == "SATtxt":
        best = best_image_to_text_match(agent)
        if best:
            return f"最匹配文本: {best['text']} (score={best['score']:.4f})"
    if agent.get("model") == "DOFA":
        dist = format_class_distribution(agent.get("class_distribution"))
        return f"DOFA 分割完成。{dist}"
    if "num_detections" in agent:
        return f"检测目标数量: {agent.get('num_detections')}"
    if agent.get("model") == "DehazeFormer":
        return f"云雾/雾霾去除完成。输出图像: {agent.get('output_image')}"
    return json.dumps(summarize_result(result), ensure_ascii=False)[:1200]


def planning_text_from_result(result: dict[str, Any]) -> str:
    plan = result.get("plan")
    if isinstance(plan, dict):
        task_type = plan.get("task_type") or "N/A"
        reasoning = plan.get("reasoning") or "N/A"
        nodes = plan.get("nodes") or []
        agents = []
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                params = node.get("params") if isinstance(node.get("params"), dict) else {}
                agents.append(
                    f"{node.get('id', 'node')}: {node.get('agent', 'unknown')}/{params.get('task', 'N/A')}"
                )
        return f"规划: task_type={task_type}; reasoning={reasoning}; agents={'; '.join(agents) or 'N/A'}"

    task_type = result.get("task_type")
    reasoning = result.get("reasoning")
    agents_value = result.get("agents")
    if task_type or reasoning or agents_value:
        agents = []
        if isinstance(agents_value, list):
            for item in agents_value:
                if isinstance(item, dict):
                    agents.append(f"{item.get('id', 'node')}: {item.get('agent', 'unknown')}/{item.get('task', 'N/A')}")
        return f"规划: task_type={task_type or 'N/A'}; reasoning={reasoning or 'N/A'}; agents={'; '.join(agents) or 'N/A'}"

    if result.get("status") == "reference_only":
        return "规划: reference_only，未调用 planner/agent。"
    return "规划: N/A"


def prediction_image_from_result(result: dict[str, Any]) -> Path | None:
    agent = first_agent_result(result)
    for key in ["visualization", "overlay", "prediction", "compare", "output_image", "mask", "rgb"]:
        value = agent.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    return None


def prediction_mask_from_result(result: dict[str, Any]) -> Path | None:
    agent = first_agent_result(result)
    for key in ["mask", "raw_prediction"]:
        value = agent.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    return None


def segmentation_metrics_text(case: Case, result: dict[str, Any]) -> str:
    pred_path = prediction_mask_from_result(result)
    gt_path = case.gt_mask_path
    if not pred_path or not gt_path or not gt_path.exists():
        return ""
    try:
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
    except Exception as exc:
        return f"分割指标: N/A（读取 mask 失败: {type(exc).__name__}）"
    if pred.shape != gt.shape:
        return f"分割指标: N/A（预测/GT 尺寸不一致: {pred.shape} vs {gt.shape}）"

    valid = gt != 255
    if not np.any(valid):
        return "分割指标: N/A（GT 没有有效像素）"

    pred_valid = pred[valid]
    gt_valid = gt[valid]
    pixel_acc = float((pred_valid == gt_valid).mean())
    classes = sorted(set(np.unique(pred_valid).tolist()) | set(np.unique(gt_valid).tolist()))
    ious = []
    for class_id in classes:
        pred_mask = pred_valid == class_id
        gt_mask = gt_valid == class_id
        union = int(np.logical_or(pred_mask, gt_mask).sum())
        if union:
            inter = int(np.logical_and(pred_mask, gt_mask).sum())
            ious.append(inter / union)
    miou = float(np.mean(ious)) if ious else 0.0
    pred_counts = dict(zip(*np.unique(pred_valid, return_counts=True)))
    gt_counts = dict(zip(*np.unique(gt_valid, return_counts=True)))
    pred_dist = ", ".join(f"{int(k)}:{int(v)}" for k, v in sorted(pred_counts.items()))
    gt_dist = ", ".join(f"{int(k)}:{int(v)}" for k, v in sorted(gt_counts.items()))
    return (
        f"分割指标(raw id): pixel_acc={pixel_acc:.3f}, mIoU={miou:.3f}; "
        f"预测类别像素={pred_dist}; GT类别像素={gt_dist}"
    )


def ensure_cjk_font(font_path: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if font_path:
        candidates.append(Path(font_path))
    candidates += [
        DEFAULT_CJK_FONT,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallback.ttf"),
    ]
    for path in candidates:
        if path.exists() and has_cjk_glyphs(path):
            return path

    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
    try:
        print(f"CJK font not found locally; downloading {url}", flush=True)
        urllib.request.urlretrieve(url, DEFAULT_CJK_FONT)
        if DEFAULT_CJK_FONT.exists() and DEFAULT_CJK_FONT.stat().st_size > 1_000_000:
            return DEFAULT_CJK_FONT
    except Exception as exc:
        print(f"Warning: failed to download CJK font: {type(exc).__name__}: {exc}", flush=True)
    return None


def has_cjk_glyphs(path: Path) -> bool:
    try:
        from fontTools.ttLib import TTCollection, TTFont

        fonts = TTCollection(str(path)).fonts if path.suffix.lower() == ".ttc" else [TTFont(str(path), lazy=True)]
        probes = [ord("测"), ord("问"), ord("遥"), ord("船")]
        for font in fonts:
            cmap: set[int] = set()
            for table in font["cmap"].tables:
                cmap.update(table.cmap.keys())
            if all(codepoint in cmap for codepoint in probes):
                return True
    except Exception:
        name = path.name.lower()
        return any(token in name for token in ["cjk", "noto", "sourcehan", "wqy", "droid", "simhei", "uming", "ukai"])
    return False


def load_font(size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cjk_font = ensure_cjk_font(font_path)
    candidates = [
        cjk_font,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path and path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int, line_gap: int = 6) -> int:
    lines = wrap_text(text, font, width)
    if not lines:
        return 0
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_h = bbox[3] - bbox[1]
    return len(lines) * line_h + max(0, len(lines) - 1) * line_gap


def wrap_text(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in paragraph):
            current = ""
            for ch in paragraph:
                test = current + ch
                if font.getlength(test) <= width or not current:
                    current = test
                else:
                    lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        else:
            for wrapped in textwrap.wrap(paragraph, width=90):
                current = ""
                for word in wrapped.split(" "):
                    test = word if not current else current + " " + word
                    if font.getlength(test) <= width or not current:
                        current = test
                    else:
                        lines.append(current)
                        current = word
                if current:
                    lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_gap: int = 6,
) -> int:
    x, y = xy
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_h = bbox[3] - bbox[1]
    for line in wrap_text(text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def fit_image(path: Path, box: tuple[int, int], background=(248, 249, 251), allow_upscale: bool = True) -> Image.Image:
    canvas = Image.new("RGB", box, background)
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", box, (235, 238, 242))
        return img
    scale = min(box[0] / max(img.width, 1), box[1] / max(img.height, 1))
    if not allow_upscale:
        scale = min(scale, 1.0)
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    resample = Image.Resampling.NEAREST if scale >= 4 else Image.Resampling.LANCZOS
    img = img.resize(new_size, resample)
    x = (box[0] - img.width) // 2
    y = (box[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def render_case_image(
    case: Case,
    result: dict[str, Any],
    output_path: Path,
    font_path: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_font = load_font(30, font_path)
    header_font = load_font(22, font_path)
    body_font = load_font(20, font_path)
    small_font = load_font(16, font_path)

    width = 1500
    margin = 34
    gap = 24
    text_w = width - 2 * margin
    top = margin + 82
    pred_image = prediction_image_from_result(result)
    panel_specs = [
        ("测试图像", case.image_path),
        ("测试结果", pred_image),
        ("GT", case.extra_gt_image_path or case.gt_image_path),
    ]
    panel_specs = [(label, Path(path)) for label, path in panel_specs if path and Path(path).exists()]
    panel_count = len(panel_specs)
    panel_h = 560 if panel_count == 1 else 420
    panel_w = (width - 2 * margin - gap * max(panel_count - 1, 0)) // max(panel_count, 1)
    panel_block_h = panel_h + 42 if panel_count else 0
    text_start_y = top + panel_block_h + (32 if panel_count else 0)

    scratch = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(scratch)
    question = f"问题: {case.query}"
    planning = planning_text_from_result(result)
    metrics = segmentation_metrics_text(case, result)
    answer = f"模型回答: {answer_from_result(result)}"
    gt = f"标准答案/GT: {case.gt_text or 'N/A'}"
    text_total = (
        text_height(draw, question, body_font, text_w)
        + text_height(draw, planning, body_font, text_w)
        + (text_height(draw, metrics, body_font, text_w) if metrics else 0)
        + text_height(draw, answer, body_font, text_w)
        + text_height(draw, gt, body_font, text_w)
        + (180 if metrics else 150)
    )
    height = text_start_y + text_total + margin
    canvas = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), f"{case.task} / {case.case_id}", font=title_font, fill=(23, 31, 42))
    header_query = f"Query: {case.query}"
    header_query_y = margin + 42
    draw_wrapped(draw, (margin, header_query_y), header_query, small_font, (67, 81, 99), text_w)

    for idx, (label, path) in enumerate(panel_specs):
        x = margin + idx * (panel_w + gap)
        draw.rounded_rectangle((x, top, x + panel_w, top + panel_h + 42), radius=8, fill=(255, 255, 255), outline=(216, 222, 230))
        draw.text((x + 16, top + 12), label, font=header_font, fill=(30, 41, 59))
        img_top = top + 42
        panel_img = fit_image(path, (panel_w - 20, panel_h - 10))
        canvas.paste(panel_img, (x + 10, img_top + 5))

    y = text_start_y
    draw.rounded_rectangle((margin, y - 16, width - margin, height - margin), radius=8, fill=(255, 255, 255), outline=(216, 222, 230))
    y += 8
    y = draw_wrapped(draw, (margin + 20, y), question, body_font, (22, 32, 45), text_w - 40)
    y += 18
    y = draw_wrapped(draw, (margin + 20, y), planning, body_font, (22, 32, 45), text_w - 40)
    y += 18
    if metrics:
        y = draw_wrapped(draw, (margin + 20, y), metrics, body_font, (22, 32, 45), text_w - 40)
        y += 18
    y = draw_wrapped(draw, (margin + 20, y), answer, body_font, (22, 32, 45), text_w - 40)
    y += 18
    y = draw_wrapped(draw, (margin + 20, y), gt, body_font, (22, 32, 45), text_w - 40)
    status = f"status={result.get('status', 'unknown')}"
    draw.text((margin + 20, height - margin - 28), status, font=small_font, fill=(95, 108, 125))
    canvas.save(output_path)


def write_reports(records: list[dict[str, Any]], output_dir: Path) -> None:
    summary_jsonl = output_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fields = [
        "case_id",
        "task",
        "status",
        "image",
        "result_image",
        "gt_image",
        "report_image",
        "answer",
        "planning",
        "metrics",
        "gt_text",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec.get(key, "") for key in fields})

    rows = []
    for rec in records:
        report_rel = Path(rec["report_image"]).relative_to(output_dir)
        rows.append(
            "<tr>"
            f"<td>{html.escape(rec['case_id'])}</td>"
            f"<td>{html.escape(rec['task'])}</td>"
            f"<td>{html.escape(rec['status'])}</td>"
            f"<td><a href='{html.escape(str(report_rel))}'><img src='{html.escape(str(report_rel))}'></a></td>"
            "</tr>"
        )
    index = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Satellite Model Dataset Test Report</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f5f7fa; color: #17202c; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #d8dee8; padding: 10px; text-align: left; vertical-align: top; }}
img {{ max-width: 520px; height: auto; display: block; }}
</style>
</head>
<body>
<h1>Satellite Model Dataset Test Report</h1>
<p>Cases: {len(records)}</p>
<table>
<thead><tr><th>Case</th><th>Task</th><th>Status</th><th>Report Image</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset_test cases and render readable image reports.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["skyeyegpt", "sarmae_detect", "mtp_dota", "sarmae_segment", "dofa_hdf5", "dofa_png_val", "sattxt"],
        choices=[
            "skyeyegpt",
            "dehazeformer",
            "sarmae_detect",
            "mtp_dota",
            "mtp_dota_object_vqa",
            "sarmae_segment",
            "sarmae_segment_raw_air",
            "dofa_hdf5",
            "dofa_png_val",
            "sattxt",
        ],
    )
    parser.add_argument("--max-cases", type=int, default=3, help="Per task. Use 0 for all cases.")
    parser.add_argument("--font", default=None, help="Optional CJK font path for report text rendering.")
    parser.add_argument("--use-test-agent-cli", action="store_true", help="Call model_wrappers/test_agent_chain.py; this also uses the API planner.")
    parser.add_argument("--fixed-planner", action="store_true", help="Debug only: bypass API planning with fixed per-task planner JSON.")
    parser.add_argument("--reuse-raw", action="store_true", help="Reuse existing raw_results/*.json in --output-dir and only rerender reports.")
    parser.add_argument("--reuse-raw-only", action="store_true", help="Only rerender cases that already have raw_results/*.json in --output-dir.")
    parser.add_argument(
        "--skyeyegpt-refer-choice",
        default="all",
        choices=["all"] + [str(item["choice"]) for item in SKYEYE_DEMO_REFERRING_QUESTIONS],
        help="For SkyEyeGPT demo referring questions, run all presets or one selected preset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List cases without running models.")
    return parser.parse_args()


def main() -> None:
    global DATASET_ROOT
    args = parse_args()
    DATASET_ROOT = args.dataset_root
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(set(args.tasks), args.max_cases, skyeyegpt_refer_choice=args.skyeyegpt_refer_choice)
    if args.reuse_raw_only:
        raw_dir_for_filter = output_dir / "raw_results"
        existing = {path.stem for path in raw_dir_for_filter.glob("*.json")}
        cases = [case for case in cases if case.case_id in existing]
        args.reuse_raw = True
    if args.dry_run:
        for case in cases:
            print(f"{case.case_id}\t{case.task}\t{case.image_path or case.input_path}")
        print(f"Total cases: {len(cases)}")
        return

    records: list[dict[str, Any]] = []
    raw_dir = output_dir / "raw_results"
    report_dir = output_dir / "case_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running {case.case_id} ({case.task})", flush=True)
        raw_path = raw_dir / f"{case.case_id}.json"
        if args.reuse_raw and raw_path.exists():
            result = load_json(raw_path)
            result["query"] = case.query
        else:
            try:
                result = (
                    run_case_cli(case)
                    if args.use_test_agent_cli
                    else run_case_direct(case, output_dir, use_fixed_planner=args.fixed_planner)
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "query": case.query,
                    "answer": f"Case failed before report rendering: {type(exc).__name__}: {exc}",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "agent_results": {},
                }
            raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        report_path = report_dir / f"{case.case_id}.png"
        render_case_image(case, result, report_path, font_path=args.font)
        pred_path = prediction_image_from_result(result)
        record = {
            "case_id": case.case_id,
            "task": case.task,
            "status": str(result.get("status", "unknown")),
            "image": str(case.image_path or ""),
            "result_image": str(pred_path or ""),
            "gt_image": str(case.extra_gt_image_path or case.gt_image_path or ""),
            "report_image": str(report_path),
            "raw_result": str(raw_path),
            "answer": answer_from_result(result),
            "planning": planning_text_from_result(result),
            "metrics": segmentation_metrics_text(case, result),
            "gt_text": case.gt_text,
        }
        records.append(record)

    write_reports(records, output_dir)
    print(f"Done. Report directory: {output_dir}")
    print(f"Open: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
