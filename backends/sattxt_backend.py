from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_DIR = Path("/root/autodl-tmp/sattxt")
WEIGHTS_DIR = PROJECT_DIR / "weights"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "thirdparty" / "dinov3"))

from sattxt.model import SATtxt  # noqa: E402
from sattxt.utils import get_preprocess, image_loader, zero_shot_classify  # noqa: E402


def ensure_local_llm2vec_adapter() -> Path:
    text_encoder_dir = WEIGHTS_DIR / "llm2vec_check"
    adapter_dir = text_encoder_dir / "unsup-simcse"
    expected_adapter_path = Path(str(text_encoder_dir) + "-unsup-simcse")
    if not expected_adapter_path.exists():
        expected_adapter_path.symlink_to(adapter_dir, target_is_directory=True)
    return text_encoder_dir


def build_model(device: str):
    os.environ.setdefault("HF_HOME", str(WEIGHTS_DIR / ".cache"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = SATtxt(
        dinov3_weights_path=str(WEIGHTS_DIR / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
        sattxt_vision_head_pretrain_weights=str(WEIGHTS_DIR / "sattxt_vision_head.pt"),
        text_encoder_id=str(ensure_local_llm2vec_adapter()),
        sattxt_text_head_pretrain_weights=str(WEIGHTS_DIR / "sattxt_text_head.pt"),
    ).to(device).eval()
    return model


@torch.no_grad()
def encode_images(model, image_paths: list[str], device: str) -> torch.Tensor:
    preprocess = get_preprocess(is_ms=False, all_bands=False)
    tensors = [preprocess(image_loader(path)).unsqueeze(0) for path in image_paths]
    batch = torch.cat(tensors, dim=0).to(device)
    dummy_captions = ["a satellite image"] * len(image_paths)
    feats, _ = model({"image": batch, "caption": dummy_captions})
    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def encode_texts(model, texts: list[str], device: str) -> torch.Tensor:
    dummy = torch.zeros((len(texts), 3, 224, 224), device=device)
    _, feats = model({"image": dummy, "caption": texts})
    return F.normalize(feats.float(), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())

    task = req["task"]
    image_paths = req.get("image_paths") or []
    texts = req.get("texts") or []
    categories = req.get("categories") or []
    output_dir = Path(req["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = req.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    result: dict
    if task == "zero_shot_classification":
        if len(image_paths) != 1:
            raise ValueError("zero_shot_classification requires exactly one image path")
        if not categories:
            raise ValueError("zero_shot_classification requires categories")
        image = image_loader(image_paths[0])
        image_tensor = get_preprocess(is_ms=False, all_bands=False)(image).unsqueeze(0).to(device)
        logits, pred_idx = zero_shot_classify(model, image_tensor, categories)
        scores = [
            {"category": category, "score": float(score)}
            for category, score in zip(categories, logits.squeeze(0).detach().cpu().tolist())
        ]
        result = {
            "model": "SATtxt",
            "task": task,
            "image": image_paths[0],
            "prediction": categories[int(pred_idx.item())],
            "scores": scores,
        }
    elif task in {"retrieval", "image_to_text", "text_to_image"}:
        if not image_paths or not texts:
            raise ValueError(f"{task} requires image_paths and texts")
        image_feats = encode_images(model, image_paths, device)
        text_feats = encode_texts(model, texts, device)
        sim = image_feats @ text_feats.T
        matrix = sim.detach().cpu().tolist()
        image_to_text = []
        for row, image_path in zip(matrix, image_paths):
            ranked = sorted(
                [{"text": text, "score": float(score)} for text, score in zip(texts, row)],
                key=lambda item: item["score"],
                reverse=True,
            )
            image_to_text.append({"image": image_path, "ranking": ranked})
        text_to_image = []
        for text_idx, text in enumerate(texts):
            ranked = sorted(
                [{"image": image_path, "score": float(matrix[i][text_idx])} for i, image_path in enumerate(image_paths)],
                key=lambda item: item["score"],
                reverse=True,
            )
            text_to_image.append({"text": text, "ranking": ranked})
        result = {
            "model": "SATtxt",
            "task": task,
            "image_paths": image_paths,
            "texts": texts,
            "similarity": matrix,
            "image_to_text": image_to_text,
            "text_to_image": text_to_image,
        }
    else:
        raise ValueError(f"Unsupported SATtxt task: {task}")

    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
