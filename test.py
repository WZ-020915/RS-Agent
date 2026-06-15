import sys
sys.path.insert(0, "/root/autodl-tmp")

from model_wrappers import skyeyegpt, sattxt

image_path = "/root/autodl-tmp/DOFA-master/datasets/detect_DOFA/JPEGImages-test/14629.jpg"

# 1. SkyEyeGPT 生成 caption
caption_result = skyeyegpt(
    image_path=image_path,
    task="caption",
    prompt="Give a concise caption in one complete sentence, under 30 words.",
    max_new_tokens=96,
    min_new_tokens=0,
    temperature=0.2,
    timeout=180,
)

caption = caption_result["answer"]
print("Caption:", caption)
print("Caption tokens:", caption_result.get("generated_tokens"))

# 2. SATtxt 零样本分类
class_result = sattxt(
    task="zero_shot_classification",
    image_paths=image_path,
    categories=[
        "industrial area",
        "residential area",
        "forest",
        "farmland",
        "river",
        "airport",
        "port",
    ],
    timeout=1200,
)

print("Prediction:", class_result["prediction"])
print("Scores:", class_result["scores"])
