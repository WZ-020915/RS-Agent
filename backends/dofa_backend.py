from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
import rasterio


PROJECT_ROOT = Path("/root/autodl-tmp/DOFA-master")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "downstream_tasks"))

from export_seg_visuals import colorize, overlay, percentile_rgb, predict_full_image, save_compare  # noqa: E402
from train_seg_dofa import (  # noqa: E402
    DofaSpatialEncoder,
    UPerLiteHead,
    WAVELENGTHS,
    actual_band_key,
    canonical_band_name,
    load_partition,
    resolve_hdf5,
    usable_band_keys,
)


HEAD_ROOT = PROJECT_ROOT / "outputs" / "dofa_seg_rgb_ms"
DATA_ROOT = PROJECT_ROOT / "datasets" / "seg_rgb-ms"
BACKBONE = PROJECT_ROOT / "checkpoints" / "DOFA_ViT_base_e100.pth"
ALLOWED_DATASETS = {
    "m-NeonTree",
    "m-SA-crop-type",
    "m-cashew-plant",
    "m-chesapeake",
    "m-nz-cattle",
    "m-pv4ger-seg",
}
DATASET_CLASS_NAMES = {
    "m-pv4ger-seg": ["background", "photovoltaic_panel"],
    "m-chesapeake": ["no_data", "water", "tree_canopy", "low_vegetation", "barren", "impervious", "building"],
}


def load_hdf5_sample(path: Path, dataset_dir: Path, band_names: list[str]) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]:
    band_stats = json.loads((dataset_dir / "band_stats.json").read_text())
    channels = []
    rgb_channels = {}
    label = None
    with h5py.File(path, "r") as f:
        label_keys = [k for k in f.keys() if k.startswith("label")]
        if label_keys:
            label = f[label_keys[0]][()].astype(np.int64)
        for band_name in band_names:
            key = actual_band_key(f, band_name)
            arr = f[key][()].astype(np.float32)
            canonical = canonical_band_name(key)
            if canonical in {"Red", "Green", "Blue"}:
                rgb_channels[canonical] = arr
            stats = band_stats[band_name]
            channels.append((arr - float(stats["mean"])) / max(float(stats["std"]), 1e-6))
    image = torch.from_numpy(np.stack(channels, axis=0)).float()
    rgb = percentile_rgb(rgb_channels) if {"Red", "Green", "Blue"} <= set(rgb_channels) else None
    return image, label, rgb


def load_png_sample(path: Path, dataset_dir: Path, band_names: list[str]) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]:
    available = {"Red", "Green", "Blue"}
    missing = [band_name for band_name in band_names if band_name not in available]
    if missing:
        raise ValueError(
            "PNG/JPEG/BMP DOFA input only provides RGB channels, but dataset "
            f"requires bands {band_names}. Missing: {missing}. Use an HDF5 sample "
            "for multispectral heads, or choose an RGB-only dataset/head."
        )

    band_stats = json.loads((dataset_dir / "band_stats.json").read_text())
    rgb = np.array(Image.open(path).convert("RGB"))
    channel_arrays = {
        "Red": rgb[:, :, 0].astype(np.float32),
        "Green": rgb[:, :, 1].astype(np.float32),
        "Blue": rgb[:, :, 2].astype(np.float32),
    }
    channels = []
    for band_name in band_names:
        arr = channel_arrays[band_name]
        stats = band_stats[band_name]
        if float(stats.get("max", 255.0)) <= 1.5:
            arr = arr / 255.0
        channels.append((arr - float(stats["mean"])) / max(float(stats["std"]), 1e-6))
    image = torch.from_numpy(np.stack(channels, axis=0)).float()
    return image, None, rgb


def normalize_channels(arrays: list[np.ndarray], dataset_dir: Path, band_names: list[str]) -> torch.Tensor:
    band_stats = json.loads((dataset_dir / "band_stats.json").read_text())
    channels = []
    for arr, band_name in zip(arrays, band_names):
        arr = arr.astype(np.float32)
        stats = band_stats[band_name]
        if float(stats.get("max", 255.0)) <= 1.5 and np.nanmax(arr) > 1.5:
            arr = arr / 255.0
        channels.append((arr - float(stats["mean"])) / max(float(stats["std"]), 1e-6))
    return torch.from_numpy(np.stack(channels, axis=0)).float()


def load_tiff_sample(path: Path, dataset_dir: Path, band_names: list[str]) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"TIFF input must be a band-first image, got shape {data.shape} from {path}")

    required_count = len(band_names)
    rgb_band_names = {"Red", "Green", "Blue"}
    if set(band_names) <= rgb_band_names and data.shape[0] >= 3:
        channel_arrays = {
            "Red": data[0],
            "Green": data[1],
            "Blue": data[2],
        }
        rgb = percentile_rgb(channel_arrays)
        return normalize_channels([channel_arrays[name] for name in band_names], dataset_dir, band_names), None, rgb

    if data.shape[0] == required_count:
        band_arrays = [data[i] for i in range(required_count)]
        rgb_channels = {}
        for arr, band_name in zip(band_arrays, band_names):
            canonical = canonical_band_name(band_name)
            if canonical in {"Red", "Green", "Blue"}:
                rgb_channels[canonical] = arr
        rgb = percentile_rgb(rgb_channels) if {"Red", "Green", "Blue"} <= set(rgb_channels) else None
        return normalize_channels(band_arrays, dataset_dir, band_names), None, rgb

    raise ValueError(
        f"TIFF input has {data.shape[0]} channels, but dataset {dataset_dir.name} "
        f"requires {required_count} bands: {band_names}. Use a TIFF with matching "
        "band count/order or an HDF5 sample with named bands."
    )


def load_sample(path: Path, dataset_dir: Path, band_names: list[str]) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return load_tiff_sample(path, dataset_dir, band_names)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        return load_png_sample(path, dataset_dir, band_names)
    return load_hdf5_sample(path, dataset_dir, band_names)


def infer_num_classes_from_checkpoint(state: dict, dataset_dir: Path) -> int:
    if "config" in state and "num_classes" in state["config"]:
        return int(state["config"]["num_classes"])
    head_state = state.get("head", state)
    for key, value in head_state.items():
        if key.endswith("fpn_bottleneck.4.weight"):
            return int(value.shape[0])
    partition = load_partition(dataset_dir, "1.00x_train_partition.json")
    max_label = 0
    for split_ids in partition.values():
        for sample_id in split_ids[:50]:
            with h5py.File(resolve_hdf5(dataset_dir, sample_id), "r") as f:
                label_key = next(k for k in f.keys() if k.startswith("label"))
                max_label = max(max_label, int(np.nanmax(f[label_key][()])))
    return max_label + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())

    dataset = req["dataset"]
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported DOFA dataset/head {dataset!r}; choices={sorted(ALLOWED_DATASETS)}")

    dataset_dir = DATA_ROOT / dataset
    head_checkpoint = HEAD_ROOT / dataset / "best.pth"
    output_dir = Path(req["output_dir"]) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    partition = load_partition(dataset_dir, "1.00x_train_partition.json")
    split = req.get("split") or "test"
    if req.get("input_path"):
        sample_path = Path(req["input_path"])
        sample_id = sample_path.stem
    else:
        sample_id = req.get("sample_id") or partition[split][0]
        sample_path = resolve_hdf5(dataset_dir, sample_id)

    first_path = resolve_hdf5(dataset_dir, partition[split][0])
    band_names = [canonical_band_name(key) for key in usable_band_keys(first_path)]
    wavelengths = [WAVELENGTHS[name] for name in band_names]

    device = torch.device(req.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(head_checkpoint, map_location=device)
    num_classes = infer_num_classes_from_checkpoint(state, dataset_dir)
    encoder = DofaSpatialEncoder(BACKBONE, "base").to(device)
    head = UPerLiteHead(encoder.out_channels, num_classes, channels=512).to(device)
    head.load_state_dict(state["head"])
    head.eval()

    image, label, rgb = load_sample(sample_path, dataset_dir, band_names)
    pred = predict_full_image(
        encoder,
        head,
        image,
        wavelengths,
        crop_size=int(req.get("crop_size") or 224),
        stride=int(req.get("stride") or 112),
        batch_size=int(req.get("batch_size") or 16),
        device=device,
    )

    safe_name = sample_id.replace("/", "_")
    raw_path = output_dir / f"{safe_name}_raw_prediction.png"
    pred_path = output_dir / f"{safe_name}_prediction.png"
    Image.fromarray(pred).save(raw_path)
    Image.fromarray(colorize(pred)).save(pred_path)
    classes, counts = np.unique(pred, return_counts=True)
    total_pixels = int(pred.size)
    class_names = DATASET_CLASS_NAMES.get(dataset, [f"class_{idx}" for idx in range(num_classes)])
    class_distribution = []
    for class_id, count in zip(classes, counts):
        class_id_int = int(class_id)
        class_distribution.append(
            {
                "class_id": class_id_int,
                "class_name": class_names[class_id_int] if class_id_int < len(class_names) else f"class_{class_id_int}",
                "pixels": int(count),
                "ratio": float(count) / max(total_pixels, 1),
            }
        )
    result = {
        "model": "DOFA",
        "task": "segmentation",
        "dataset": dataset,
        "sample_id": sample_id,
        "input_path": str(sample_path),
        "head_checkpoint": str(head_checkpoint),
        "num_classes": num_classes,
        "class_names": class_names,
        "class_pixel_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "class_distribution": class_distribution,
        "raw_prediction": str(raw_path),
        "prediction": str(pred_path),
    }
    if rgb is not None:
        rgb_path = output_dir / f"{safe_name}_rgb.png"
        overlay_path = output_dir / f"{safe_name}_overlay.png"
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(overlay(rgb, pred)).save(overlay_path)
        result.update({"rgb": str(rgb_path), "overlay": str(overlay_path)})
    if label is not None and rgb is not None:
        compare_path = output_dir / f"{safe_name}_compare.png"
        save_compare(compare_path, rgb, label, pred)
        result["compare"] = str(compare_path)

    (output_dir / f"{safe_name}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
