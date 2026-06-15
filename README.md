# Remote Sensing Model Wrappers

This package exposes four callable functions:

- `skyeyegpt(...)`: SkyEyeGPT chat / grounding / caption / VQA.
- `sarmae(...)`: SARMAE SAR target detection and SAR semantic segmentation.
- `mtp(...)`: MTP optical remote-sensing object detection, including horizontal boxes and rotated boxes.
- `dofa(...)`: DOFA segmentation using one of the six heads in `DOFA-master/outputs/dofa_seg_rgb_ms`.
- `sattxt(...)`: SATtxt zero-shot classification and image-text retrieval.
- `run_remote_sensing_agent_system(...)`: LLM-planned multi-agent orchestration over the model wrappers.

The public functions run backend scripts in the required conda environments:

| Function | Conda env | Backend |
| --- | --- | --- |
| `skyeyegpt` | `minigptv` | `backends/skyeyegpt_backend.py` |
| `sarmae(..., task="detect")` | `sarmae` | `backends/sarmae_backend.py` |
| `sarmae(..., task="segment")` | `sarmae_seg` | `backends/sarmae_backend.py` |
| `mtp` | `mtp` | `backends/mtp_backend.py` |
| `dofa` | `dofa` | `backends/dofa_backend.py` |
| `sattxt` | `sattxt` | `backends/sattxt_backend.py` |

## Examples

```python
import sys
sys.path.insert(0, "/root/autodl-tmp")

from model_wrappers import dofa, mtp, sarmae, sattxt, skyeyegpt

# DOFA segmentation, using the first test sample from one of the six heads.
print(dofa("m-pv4ger-seg"))

# SARMAE SSDD target detection.
print(sarmae(
    "/root/autodl-tmp/SARMAE-main/Official-SSDD-OPEN/BBox_SSDD/coco_style/images/test/000001.jpg",
    task="detect",
))

# SARMAE SAR semantic segmentation.
print(sarmae(
    "/root/autodl-tmp/SARMAE-main/Raw_AIR-PolarSAR-Seg/test_set/images/AIR-PolarSAR-Seg-100_HH.png",
    task="segment",
))

# SARMAE detection and segmentation in one call.
print(sarmae(
    "/root/autodl-tmp/SARMAE-main/Raw_AIR-PolarSAR-Seg/test_set/images/AIR-PolarSAR-Seg-100_HH.png",
    task="both",
))

# MTP DIOR horizontal object detection.
print(mtp(
    "/root/autodl-tmp/DOFA-master/datasets/DIOR/JPEGImages-test/11726.jpg",
    task="horizontal",
))

# MTP DIOR-R rotated object detection.
print(mtp(
    "/root/autodl-tmp/DOFA-master/datasets/DIOR/JPEGImages-test/11726.jpg",
    task="rotated",
))

# Run both MTP detectors in one call.
print(mtp(
    "/root/autodl-tmp/DOFA-master/datasets/DIOR/JPEGImages-test/11726.jpg",
    task="both",
))

# SATtxt zero-shot classification.
print(sattxt(
    task="zero_shot_classification",
    image_paths="/root/autodl-tmp/sattxt/asset/Residential_167.jpg",
    categories=["AnnualCrop", "Forest", "Residential", "River", "SeaLake"],
))

# SATtxt retrieval.
print(sattxt(
    task="retrieval",
    image_paths=["/root/autodl-tmp/sattxt/asset/Residential_167.jpg"],
    texts=["a residential satellite image", "a forest satellite image"],
))

# SkyEyeGPT caption / VQA / grounding.
print(skyeyegpt(
    image_path="/path/to/image.jpg",
    task="caption",
    prompt="Give a concise caption in one complete sentence, under 30 words.",
    max_new_tokens=96,
    min_new_tokens=0,
    temperature=0.2,
))

# SkyEyeGPT multi-turn context. Pass the previous updated_history into the next call.
turn1 = skyeyegpt(
    image_path="/path/to/image.jpg",
    task="vqa",
    prompt="What objects are visible in this remote sensing image?",
)
turn2 = skyeyegpt(
    image_path="/path/to/image.jpg",
    task="vqa",
    prompt="Which of those objects are closest to the road?",
    history=turn1["updated_history"],
)
print(turn2["answer"])
```

## Multi-Agent Orchestration

The orchestration layer lives in `agent_system.py`. It does four things:

1. Calls an LLM planner to return JSON.
2. Validates the JSON and converts it to an executable dependency graph.
3. Executes `skyeyegpt`, `sarmae`, `dofa`, and/or `sattxt` by graph level. Independent nodes run in parallel.
4. Calls an LLM synthesizer to turn agent results into a human-readable answer. If no API key is configured and a manual plan is supplied, it uses a deterministic fallback summary.

```python
from model_wrappers import run_remote_sensing_agent_system

result = run_remote_sensing_agent_system(
    "图中有几个船？",
    image_path="/root/autodl-tmp/SARMAE-main/Official-SSDD-OPEN/BBox_SSDD/coco_style/images/test/000001.jpg",
)
print(result["answer"])
```

The default LLM client uses the OpenAI Python SDK with DMXAPI Responses API:

- `DMXAPI_API_KEY` or `OPENAI_API_KEY`
- `DMXAPI_BASE_URL` or `OPENAI_BASE_URL`, optional, default `https://www.dmxapi.cn/v1`
- `REMOTE_SENSING_LLM_MODEL`, optional, default `gpt-5.4`

Environment variables are loaded with `python-dotenv` from `/root/autodl-tmp/key.env`, `/root/autodl-tmp/.env`, or from a path set by `REMOTE_SENSING_ENV_FILE`. The code does not create env files.

Recommended `/root/autodl-tmp/key.env` format:

```bash
DMXAPI_API_KEY=sk-...
DMXAPI_BASE_URL=https://www.dmxapi.cn/v1
REMOTE_SENSING_LLM_MODEL=gpt-5.4
```

For convenience, `key.env` may also contain only the raw `sk-...` token on one line; it will be treated as `DMXAPI_API_KEY`.

The built-in call path is equivalent to:

```python
from openai import OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://www.dmxapi.cn/v1",
)
response = client.responses.create(
    model="gpt-5.4",
    input="...",
)
```

You can still replace the GPT5.4 API piece by passing custom callables:

```python
def gpt54_planner(context):
    # Call GPT5.4 and return the parsed planner JSON.
    return {
        "task_type": "counting",
        "reasoning": "SAR 目标计数先检测，再由最终回答统计检测框数量。",
        "nodes": [
            {
                "id": "detect_targets",
                "agent": "sarmae",
                "depends_on": [],
                "params": {"task": "detect", "image_path": "${context.image_path}"},
            }
        ],
    }

def gpt54_synthesizer(context, plan, results):
    # Call GPT5.4 again and return final natural-language answer.
    return f"图中有 {results['detect_targets']['num_detections']} 个目标。"

result = run_remote_sensing_agent_system(
    "图中有几个船？",
    image_path="/path/to/sar.jpg",
    planner=gpt54_planner,
    synthesizer=gpt54_synthesizer,
)
```

For debugging graph conversion without any API call:

```python
manual_plan = {
    "task_type": "caption",
    "reasoning": "图像描述由 SkyEyeGPT 完成。",
    "nodes": [
        {"id": "caption", "agent": "skyeyegpt", "depends_on": [], "params": {"task": "caption"}}
    ],
}

planned = run_remote_sensing_agent_system(
    "描述这张图",
    image_path="/path/to/image.jpg",
    planner_json=manual_plan,
    execute=False,
)
print(planned["graph"])
```

CLI:

```bash
python model_wrappers/run_agent_system.py "图中有几个船？" \
  --image-path /path/to/sar.jpg
```

## Notes

- SARMAE detection uses `/root/autodl-tmp/SARMAE-main/weights/detect_epoch_34.pth`.
- SARMAE segmentation uses `/root/autodl-tmp/SARMAE-main/weights/seg_iter_20000.pth`.
- MTP horizontal detection uses `/root/autodl-tmp/MTP-main/weights/dior-rvsa-b-mae-mtp-epoch_12.pth`.
- MTP rotated detection uses `/root/autodl-tmp/MTP-main/weights/diorr-rvsa-b-mae-mtp-epoch_12.pth`.
- DOFA uses `/root/autodl-tmp/DOFA-master/checkpoints/DOFA_ViT_base_e100.pth` plus the selected `best.pth` segmentation head.
- SARMAE detection uses the `sarmae` environment with the mmrotate stack. SARMAE segmentation uses the separate `sarmae_seg` environment with the mmsegmentation/mmengine stack and `mmcv>=2`.
- The `skyeyegpt` conda environment is incomplete on this machine, so the SkyEyeGPT wrapper runs in the existing `minigptv` environment, which has the MiniGPT-v2 runtime dependencies installed.
