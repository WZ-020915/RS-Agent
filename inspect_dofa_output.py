from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {
    "m-pv4ger-seg": ["background", "photovoltaic_panel"],
    "m-chesapeake": ["no_data", "water", "tree_canopy", "low_vegetation", "barren", "impervious", "building"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DOFA segmentation output directory.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/root/autodl-tmp/model_wrappers/outputs/agent_system/seg/m-pv4ger-seg",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    result_files = sorted(output_dir.glob("*_result.json"))
    if not result_files:
        raise FileNotFoundError(f"No *_result.json found in {output_dir}")

    result = json.loads(result_files[-1].read_text())
    dataset = result.get("dataset", output_dir.name)
    mask_path = Path(result["raw_prediction"])
    mask = np.asarray(Image.open(mask_path))
    classes, counts = np.unique(mask, return_counts=True)
    names = CLASS_NAMES.get(dataset, [f"class_{idx}" for idx in range(int(mask.max()) + 1)])
    distribution = []
    for class_id, count in zip(classes, counts):
        class_id_int = int(class_id)
        distribution.append(
            {
                "class_id": class_id_int,
                "class_name": names[class_id_int] if class_id_int < len(names) else f"class_{class_id_int}",
                "pixels": int(count),
                "ratio": float(count) / int(mask.size),
            }
        )

    summary = {
        "dataset": dataset,
        "sample_id": result.get("sample_id"),
        "prediction": result.get("prediction"),
        "overlay": result.get("overlay"),
        "compare": result.get("compare"),
        "raw_prediction": result.get("raw_prediction"),
        "class_distribution": distribution,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
