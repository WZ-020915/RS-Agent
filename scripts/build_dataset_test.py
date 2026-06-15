from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


SRC_ROOT = Path("/root/autodl-tmp/dataset")
DOFA_HDF5_ROOT = Path("/root/autodl-tmp/DOFA-master/datasets/seg_rgb-ms")
OUT_ROOT = Path("/root/autodl-tmp/model_wrappers/dataset_test")
N = 10

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
    ],
    dtype=np.uint8,
)


def rel(path: Path) -> str:
    return str(path.relative_to(OUT_ROOT))


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return rel(dst)


def select(paths: list[Path], n: int = N) -> list[Path]:
    return sorted(paths)[:n]


def to_label_array(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1])
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        return inv.reshape(arr.shape[:2]).astype(np.int64)
    return arr.astype(np.int64)


def colorize(mask: np.ndarray) -> np.ndarray:
    max_id = int(mask.max()) if mask.size else 0
    palette = PALETTE
    if max_id >= len(palette):
        rng = np.random.default_rng(0)
        extra = rng.integers(0, 255, size=(max_id + 1 - len(palette), 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    return palette[np.clip(mask, 0, len(palette) - 1)]


def save_mask_visuals(image_path: Path, mask_path: Path, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = to_label_array(mask_path)
    color = colorize(mask)
    image = Image.open(image_path).convert("RGB")
    if image.size != (mask.shape[1], mask.shape[0]):
        image = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    overlay = (np.array(image, dtype=np.float32) * 0.62 + color.astype(np.float32) * 0.38).astype(np.uint8)

    color_path = out_dir / f"{stem}_mask_color.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    Image.fromarray(color).save(color_path)
    Image.fromarray(overlay).save(overlay_path)
    classes, counts = np.unique(mask, return_counts=True)
    return {
        "mask_color": rel(color_path),
        "overlay": rel(overlay_path),
        "class_pixel_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }


def sample_skyeyegpt(manifest: dict) -> None:
    out = OUT_ROOT / "skyeyegpt"
    images = select(list((SRC_ROOT / "skyeyegpt" / "Images_val").glob("*.png")))
    records = []
    for image in images:
        records.append(
            {
                "sample_id": image.stem,
                "image": copy_file(image, out / "images" / image.name),
                "task_hint": "caption/vqa/grounding",
            }
        )
    for ann in sorted((SRC_ROOT / "skyeyegpt").glob("*.json")):
        copy_file(ann, out / ann.name)
    manifest["skyeyegpt"] = {"count": len(records), "records": records}


def sample_sattxt(manifest: dict) -> None:
    src = SRC_ROOT / "sattxt"
    out = OUT_ROOT / "sattxt"
    images = select([p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS]) if src.exists() else []
    records = []
    for image in images:
        records.append({"sample_id": image.stem, "image": copy_file(image, out / "images" / image.name)})
    manifest["sattxt"] = {
        "count": len(records),
        "records": records,
        "note": "No files were present under /root/autodl-tmp/dataset/sattxt." if not records else "",
    }


def sample_sarmae_detect(manifest: dict) -> None:
    out = OUT_ROOT / "sarmae_detect"
    src = SRC_ROOT / "sarmae" / "HRSID_JPG"
    images = select(list((src / "JPEGImages").glob("*.jpg")))
    records = []
    for image in images:
        records.append(
            {
                "sample_id": image.stem,
                "image": copy_file(image, out / "images" / image.name),
                "task_hint": "detect",
            }
        )
    for ann in sorted((src / "annotations").glob("*.json")):
        copy_file(ann, out / "annotations" / ann.name)
    manifest["sarmae_detect"] = {"count": len(records), "records": records}


def sample_sarmae_segment(manifest: dict) -> None:
    out = OUT_ROOT / "sarmae_segment"
    img_dir = SRC_ROOT / "sarmae" / "seg" / "test-00000-of-00001_extracted" / "images"
    mask_dir = SRC_ROOT / "sarmae" / "seg" / "test-00000-of-00001_extracted" / "labels"
    images = select([p for p in img_dir.glob("*.png") if (mask_dir / p.name).exists()])
    records = []
    for image in images:
        mask = mask_dir / image.name
        item = {
            "sample_id": image.stem,
            "image": copy_file(image, out / "images" / image.name),
            "mask": copy_file(mask, out / "masks" / mask.name),
            "task_hint": "segment",
        }
        item.update(save_mask_visuals(image, mask, out / "visuals", image.stem))
        records.append(item)
    manifest["sarmae_segment"] = {"count": len(records), "records": records}


def sample_mtp(manifest: dict) -> None:
    out = OUT_ROOT / "mtp_dota"
    root = SRC_ROOT / "mtp" / "DOTAv1.0"
    candidates = []
    for split in ["val", "train"]:
        for image in sorted((root / split / "images").glob("*.png")):
            label = root / "labels" / split / f"{image.stem}.txt"
            if label.exists():
                candidates.append((split, image, label))
            if len(candidates) >= N:
                break
        if len(candidates) >= N:
            break
    records = []
    for split, image, label in candidates:
        records.append(
            {
                "sample_id": image.stem,
                "split": split,
                "image": copy_file(image, out / split / "images" / image.name),
                "label": copy_file(label, out / split / "labels" / label.name),
                "task_hint": "rotated_detection",
            }
        )
    text_json = root / "labels" / "dota_seen_class_texts.json"
    if text_json.exists():
        copy_file(text_json, out / "labels" / text_json.name)
    manifest["mtp_dota"] = {"count": len(records), "records": records}


def sample_dofa_png(manifest: dict) -> None:
    out = OUT_ROOT / "dofa_png_val"
    root = SRC_ROOT / "dofa" / "Val"
    pairs = []
    for area in ["Rural", "Urban"]:
        img_dir = root / area / "images_png"
        mask_dir = root / area / "masks_png"
        for image in sorted(img_dir.glob("*.png")):
            mask = mask_dir / image.name
            if mask.exists():
                pairs.append((area, image, mask))
            if len(pairs) >= N:
                break
        if len(pairs) >= N:
            break
    records = []
    for area, image, mask in pairs:
        item = {
            "sample_id": image.stem,
            "area": area,
            "image": copy_file(image, out / area / "images" / image.name),
            "mask": copy_file(mask, out / area / "masks" / mask.name),
            "task_hint": "segment_png_reference",
        }
        item.update(save_mask_visuals(image, mask, out / area / "visuals", image.stem))
        records.append(item)
    manifest["dofa_png_val"] = {"count": len(records), "records": records}


def hdf5_rgb_and_label(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        lower = {k.lower(): k for k in keys}
        rgb_keys = []
        for names in [("red", "green", "blue"), ("r", "g", "b")]:
            if all(name in lower for name in names):
                rgb_keys = [lower[name] for name in names]
                break
        if not rgb_keys:
            band_keys = [k for k in keys if h5[k].ndim == 2 and not k.lower().startswith("label")]
            rgb_keys = band_keys[:3]
        rgb = None
        if len(rgb_keys) >= 3:
            channels = [h5[k][()].astype(np.float32) for k in rgb_keys[:3]]
            normed = []
            for arr in channels:
                lo, hi = np.nanpercentile(arr, [2, 98])
                normed.append(np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1))
            rgb = (np.stack(normed, axis=-1) * 255).astype(np.uint8)
        label = None
        label_keys = [k for k in keys if k.lower().startswith("label")]
        if label_keys:
            label = h5[label_keys[0]][()].astype(np.int64)
        return rgb, label


def sample_dofa_hdf5(manifest: dict) -> None:
    out = OUT_ROOT / "dofa_hdf5"
    records_by_dataset = {}
    for dataset_dir in sorted([p for p in DOFA_HDF5_ROOT.iterdir() if p.is_dir()]):
        records = []
        files = select(list(dataset_dir.glob("*.hdf5")) + list(dataset_dir.glob("*.h5")))
        dst_dataset = out / dataset_dir.name
        for h5_path in files:
            stem = h5_path.stem
            item = {
                "sample_id": stem,
                "dataset": dataset_dir.name,
                "hdf5": copy_file(h5_path, dst_dataset / "hdf5" / h5_path.name),
                "task_hint": "dofa_segmentation_hdf5",
                "wrapper_request_example": {
                    "dataset": dataset_dir.name,
                    "input_path": str(dst_dataset / "hdf5" / h5_path.name),
                    "output_dir": "/root/autodl-tmp/model_wrappers/outputs/dofa_dataset_test",
                },
            }
            rgb, label = hdf5_rgb_and_label(h5_path)
            if rgb is not None:
                rgb_path = dst_dataset / "visuals" / f"{stem}_rgb.png"
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(rgb_path)
                item["rgb"] = rel(rgb_path)
            if label is not None:
                raw_path = dst_dataset / "masks" / f"{stem}_label.png"
                color_path = dst_dataset / "visuals" / f"{stem}_label_color.png"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                color_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(label.astype(np.uint8)).save(raw_path)
                Image.fromarray(colorize(label)).save(color_path)
                item["label"] = rel(raw_path)
                item["label_color"] = rel(color_path)
                classes, counts = np.unique(label, return_counts=True)
                item["class_pixel_counts"] = {str(int(c)): int(n) for c, n in zip(classes, counts)}
                if rgb is not None:
                    overlay = (rgb.astype(np.float32) * 0.62 + colorize(label).astype(np.float32) * 0.38).astype(np.uint8)
                    overlay_path = dst_dataset / "visuals" / f"{stem}_overlay.png"
                    Image.fromarray(overlay).save(overlay_path)
                    item["overlay"] = rel(overlay_path)
            records.append(item)
        band_stats = dataset_dir / "band_stats.json"
        if band_stats.exists():
            copy_file(band_stats, dst_dataset / "band_stats.json")
        records_by_dataset[dataset_dir.name] = {"count": len(records), "records": records}
    manifest["dofa_hdf5"] = records_by_dataset


def main() -> None:
    ensure_clean_dir(OUT_ROOT)
    manifest: dict = {"output_root": str(OUT_ROOT), "samples_per_dataset": N}
    sample_skyeyegpt(manifest)
    sample_sattxt(manifest)
    sample_sarmae_detect(manifest)
    sample_sarmae_segment(manifest)
    sample_mtp(manifest)
    sample_dofa_png(manifest)
    sample_dofa_hdf5(manifest)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v.get("count", "grouped") if isinstance(v, dict) else "ok" for k, v in manifest.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
