from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SKYEYEGPT_CKPT = Path("/root/autodl-tmp/SkyEyeGPT/SkyEyeGPT.pth")
MINIGPT_ROOT = Path("/root/autodl-tmp/MiniGPT-4-main")
MINIGPT_CFG = MINIGPT_ROOT / "eval_configs" / "minigptv2_eval.yaml"


def normalize_history(raw_history: object) -> list[dict[str, str]]:
    if raw_history is None:
        return []
    if not isinstance(raw_history, list):
        raise ValueError("history must be a list of {'role': ..., 'content': ...} dicts or (role, content) tuples")

    normalized = []
    aliases = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "ai": "assistant",
        "model": "assistant",
    }
    for idx, item in enumerate(raw_history):
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role, content = item
        else:
            raise ValueError(f"history[{idx}] must be a dict or a two-item tuple/list")

        normalized_role = aliases.get(str(role).lower())
        if normalized_role is None:
            raise ValueError(f"history[{idx}] role must be 'user' or 'assistant', got {role!r}")
        if content is None:
            raise ValueError(f"history[{idx}] content must not be None")
        normalized.append({"role": normalized_role, "content": str(content)})

    for idx, turn in enumerate(normalized):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if turn["role"] != expected_role:
            raise ValueError(
                "history must alternate user/assistant turns and start with user; "
                f"history[{idx}] has role {turn['role']!r}, expected {expected_role!r}"
            )
    if normalized and normalized[-1]["role"] != "assistant":
        raise ValueError("history must end with an assistant turn before the current prompt")
    return normalized


def unavailable(req: dict, reason: str) -> dict:
    return {
        "model": "SkyEyeGPT",
        "task": req.get("task"),
        "status": "unavailable",
        "reason": reason,
        "checkpoint": str(SKYEYEGPT_CKPT),
        "expected_runtime": str(MINIGPT_ROOT),
        "input": {"image_path": req.get("image_path"), "prompt": req.get("prompt"), "history": req.get("history") or []},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())

    if not SKYEYEGPT_CKPT.exists():
        print(json.dumps(unavailable(req, f"Checkpoint not found: {SKYEYEGPT_CKPT}"), ensure_ascii=False))
        return
    if not MINIGPT_CFG.exists():
        print(json.dumps(unavailable(req, f"MiniGPT-v2 config not found: {MINIGPT_CFG}"), ensure_ascii=False))
        return

    sys.path.insert(0, str(MINIGPT_ROOT))
    try:
        from PIL import Image
        import torch
        from omegaconf import OmegaConf

        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2
        import minigpt4.models  # noqa: F401
        import minigpt4.processors  # noqa: F401
    except Exception as exc:
        print(
            json.dumps(
                unavailable(
                    req,
                    "SkyEyeGPT uses MiniGPT-v2 code, but the current skyeyegpt env cannot import the runtime: "
                    f"{type(exc).__name__}: {exc}",
                ),
                ensure_ascii=False,
            )
        )
        return

    task = req.get("task") or "chat"
    image_path = req.get("image_path")
    prompt = req.get("prompt") or ""
    history = normalize_history(req.get("history") or [])
    if task in {"caption", "vqa", "grounding"} and not image_path:
        raise ValueError(f"SkyEyeGPT task {task!r} requires image_path")

    try:
        cfg_dict = OmegaConf.load(MINIGPT_CFG)
        cfg_dict.model.ckpt = str(SKYEYEGPT_CKPT)
        tmp_cfg = Path(req.get("output_dir") or "/root/autodl-tmp/model_wrappers/outputs/skyeyegpt")
        tmp_cfg.mkdir(parents=True, exist_ok=True)
        cfg_path = tmp_cfg / "skyeyegpt_runtime.yaml"
        OmegaConf.save(cfg_dict, cfg_path)

        class Args:
            pass

        args_obj = Args()
        args_obj.cfg_path = str(cfg_path)
        args_obj.options = None
        args_obj.gpu_id = 0

        cfg = Config(args_obj)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to("cuda:0" if torch.cuda.is_available() else "cpu")
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device="cuda:0" if torch.cuda.is_available() else "cpu")

        chat_state = CONV_VISION_minigptv2.copy()
        img_list = []
        if image_path:
            image = Image.open(image_path).convert("RGB")
            chat.upload_img(image, chat_state, img_list)
            chat.encode_img(img_list)

        if task == "caption" and not prompt:
            prompt = "Give a concise caption in one complete sentence, under 30 words."
        elif task == "grounding" and prompt:
            prompt = f"Locate the target described as: {prompt}"

        for turn in history:
            if turn["role"] == "user":
                chat.ask(turn["content"], chat_state)
            else:
                chat_state.append_message(chat_state.roles[1], turn["content"])

        chat.ask(prompt, chat_state)
        min_new_tokens = req.get("min_new_tokens")
        generation_kwargs = chat.answer_prepare(
            conv=chat_state,
            img_list=img_list,
            num_beams=int(req.get("num_beams") or 1),
            temperature=float(req.get("temperature") or 0.6),
            max_new_tokens=int(req.get("max_new_tokens") or 512),
        )
        if min_new_tokens is not None and int(min_new_tokens) > 0:
            generation_kwargs["min_new_tokens"] = int(min_new_tokens)
        output_token = chat.model_generate(**generation_kwargs)[0]
        answer = model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        answer = answer.split("###")[0]
        answer = answer.split("Assistant:")[-1].strip()
        if task == "caption" and answer and answer[-1] not in ".!?。！？":
            answer += "."
        chat_state.messages[-1][1] = answer
        updated_history = [*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        result = {
            "model": "SkyEyeGPT",
            "task": task,
            "status": "ok",
            "image_path": image_path,
            "prompt": prompt,
            "history": history,
            "updated_history": updated_history,
            "answer": answer,
            "generated_tokens": int(output_token.numel()),
            "min_new_tokens": int(min_new_tokens) if min_new_tokens is not None else None,
            "max_new_tokens": int(req.get("max_new_tokens") or 512),
            "checkpoint": str(SKYEYEGPT_CKPT),
        }
    except Exception as exc:
        result = unavailable(req, f"SkyEyeGPT runtime failed: {type(exc).__name__}: {exc}")

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
