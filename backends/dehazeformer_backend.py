from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


DEHAZE_ROOT = Path("/root/autodl-tmp/DehazeFormer")


def unavailable(req: dict, reason: str) -> dict:
    return {
        "model": "DehazeFormer",
        "task": "dehaze",
        "status": "unavailable",
        "reason": reason,
        "image": req.get("image_path"),
    }


def resolve_checkpoint(req: dict) -> Path:
    checkpoint = req.get("checkpoint")
    if checkpoint:
        return Path(str(checkpoint))
    exp = str(req.get("exp") or "reside6k")
    model_name = str(req.get("model_name") or "dehazeformer-b")
    candidates = [
        DEHAZE_ROOT / "saved_models" / exp / f"{model_name}.pth",
        DEHAZE_ROOT / "save_models" / exp / f"{model_name}.pth",
        DEHAZE_ROOT / "saved_models" / f"{model_name}.pth",
        DEHAZE_ROOT / "save_models" / f"{model_name}.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def strip_data_parallel(state_dict: dict) -> OrderedDict:
    new_state = OrderedDict()
    for key, value in state_dict.items():
        new_state[key[7:] if key.startswith("module.") else key] = value
    return new_state


def load_input(path: Path):
    import torch

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)
    return tensor * 2.0 - 1.0, image.size


def save_output(tensor, path: Path) -> None:
    arr = tensor.detach().cpu().squeeze(0).clamp(0, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.round(arr * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())

    if not DEHAZE_ROOT.exists():
        print(json.dumps(unavailable(req, f"DehazeFormer root not found: {DEHAZE_ROOT}"), ensure_ascii=False))
        return

    image_path = Path(str(req.get("image_path") or ""))
    if not image_path.exists():
        print(json.dumps(unavailable(req, f"Input image not found: {image_path}"), ensure_ascii=False))
        return

    checkpoint = resolve_checkpoint(req)
    if not checkpoint.exists():
        print(
            json.dumps(
                unavailable(
                    req,
                    "Checkpoint not found. Expected a DehazeFormer .pth file at "
                    f"{checkpoint}; pass checkpoint=... or place weights under saved_models/<exp>/.",
                ),
                ensure_ascii=False,
            )
        )
        return

    sys.path.insert(0, str(DEHAZE_ROOT))
    import torch
    from models import dehazeformer_b, dehazeformer_d, dehazeformer_l, dehazeformer_m, dehazeformer_s, dehazeformer_t, dehazeformer_w

    model_fns = {
        "dehazeformer-t": dehazeformer_t,
        "dehazeformer-s": dehazeformer_s,
        "dehazeformer-b": dehazeformer_b,
        "dehazeformer-d": dehazeformer_d,
        "dehazeformer-w": dehazeformer_w,
        "dehazeformer-m": dehazeformer_m,
        "dehazeformer-l": dehazeformer_l,
    }
    model_name = str(req.get("model_name") or "dehazeformer-b")
    if model_name not in model_fns:
        raise ValueError(f"Unsupported DehazeFormer model_name {model_name!r}; choices={sorted(model_fns)}")

    device = torch.device(req.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    network = model_fns[model_name]().to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("state_dict", state)
    network.load_state_dict(strip_data_parallel(state_dict))
    network.eval()

    input_tensor, original_size = load_input(image_path)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = network(input_tensor).clamp_(-1, 1)
        output = output * 0.5 + 0.5

    output_dir = Path(req.get("output_dir") or "/root/autodl-tmp/model_wrappers/outputs/dehazeformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_dehazed.png"
    save_output(output, out_path)

    result = {
        "model": "DehazeFormer",
        "task": "dehaze",
        "status": "ok",
        "image": str(image_path),
        "output_image": str(out_path),
        "checkpoint": str(checkpoint),
        "model_name": model_name,
        "exp": str(req.get("exp") or "reside6k"),
        "device": str(device),
        "original_size": list(original_size),
    }
    (output_dir / f"{image_path.stem}_dehazeformer_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
