"""LLM-planned multi-agent orchestration for remote-sensing models."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .remote_models import dehazeformer, dofa, mtp, sarmae, sattxt, skyeyegpt


ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    """Load secrets from explicit/project .env files without overriding env vars."""

    env_file = os.environ.get("REMOTE_SENSING_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
    key_env = ROOT.parent / "key.env"
    load_dotenv(key_env, override=False)
    load_raw_key_file(key_env)
    load_dotenv(ROOT.parent / ".env", override=False)


def load_raw_key_file(path: Path) -> None:
    """Accept a key.env containing only the raw sk-* token."""

    if os.environ.get("DMXAPI_API_KEY") or os.environ.get("OPENAI_API_KEY") or not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        value = line.strip().strip("\"'")
        if not value or value.startswith("#") or "=" in value:
            continue
        if value.startswith("sk-"):
            os.environ["DMXAPI_API_KEY"] = value
        return


load_environment()

DEFAULT_LLM_MODEL = os.environ.get("REMOTE_SENSING_LLM_MODEL", "gpt-5.4")
DEFAULT_LLM_BASE_URL = os.environ.get("DMXAPI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/model_wrappers/outputs/agent_system"
PlannerFn = Callable[[dict[str, Any]], dict[str, Any]]
SynthesizerFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str]


class AgentSystemError(RuntimeError):
    """Raised when planning, graph validation, execution, or synthesis fails."""


class OpenAICompatibleLLM:
    """DMXAPI/OpenAI Responses client used by the planner and synthesizer."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        self.model = model or DEFAULT_LLM_MODEL
        self.api_key = api_key or os.environ.get("DMXAPI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or DEFAULT_LLM_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise AgentSystemError("DMXAPI_API_KEY or OPENAI_API_KEY is required for LLM planning and final answer synthesis")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentSystemError("The 'openai' package is required for DMXAPI calls: pip install openai") from exc

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False, temperature: float = 0.1) -> Any:
        prompt = self._messages_to_input(messages)
        if json_mode:
            prompt += "\n\n请严格只输出一个合法 JSON 对象，不要使用 Markdown 代码块，不要输出解释。"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                return self._response_to_text(response)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise AgentSystemError(f"LLM request failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _messages_to_input(messages: list[dict[str, str]]) -> str:
        chunks = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            chunks.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(chunks)

    @staticmethod
    def _response_to_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        data = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(data, dict):
            texts = []
            for item in data.get("output", []) or []:
                for content in item.get("content", []) or []:
                    if isinstance(content, dict):
                        text = content.get("text") or content.get("output_text")
                        if text:
                            texts.append(str(text))
            if texts:
                return "\n".join(texts)
            return json.dumps(data, ensure_ascii=False)
        return str(response)


def run_remote_sensing_agent_system(
    user_query: str,
    *,
    image_path: str | os.PathLike[str] | None = None,
    image_paths: list[str | os.PathLike[str]] | None = None,
    history: list[dict[str, str] | tuple[str, str]] | None = None,
    categories: list[str] | None = None,
    texts: list[str] | None = None,
    dataset: str | None = None,
    sample_id: str | None = None,
    input_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    planner_json: dict[str, Any] | None = None,
    planner: PlannerFn | None = None,
    synthesizer: SynthesizerFn | None = None,
    execute: bool = True,
    llm: OpenAICompatibleLLM | None = None,
) -> dict[str, Any]:
    """Plan, execute, and summarize a remote-sensing task.

    ``planner`` and ``synthesizer`` are the intended integration points for a
    GPT5.4 API adapter. ``planner_json`` skips planning entirely for debugging.
    """

    context = {
        "user_query": user_query,
        "image_path": str(image_path) if image_path is not None else None,
        "image_paths": [str(p) for p in image_paths] if image_paths else ([str(image_path)] if image_path else []),
        "categories": categories or [],
        "texts": texts or [],
        "dataset": dataset,
        "sample_id": sample_id,
        "input_path": str(input_path) if input_path is not None else None,
        "history": history or [],
        "output_dir": str(output_dir),
    }
    llm_client = llm
    plan = planner_json
    if plan is None:
        if planner is not None:
            plan = planner(context)
        else:
            llm_client = llm_client or OpenAICompatibleLLM()
            plan = plan_with_llm(llm_client, context)
    graph = build_execution_graph(plan)

    if not execute:
        return {"status": "planned", "plan": plan, "graph": graph}

    agent_results = execute_graph(graph, context)
    if synthesizer is not None:
        final_answer = synthesizer(context, plan, agent_results)
    elif llm_client is not None or os.environ.get("DMXAPI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        llm_client = llm_client or OpenAICompatibleLLM()
        final_answer = synthesize_answer(llm_client, context, plan, agent_results)
    else:
        final_answer = synthesize_answer_without_llm(context, plan, agent_results)
    return {
        "status": "ok",
        "query": user_query,
        "plan": plan,
        "graph": graph,
        "agent_results": agent_results,
        "answer": final_answer,
    }


def plan_with_llm(llm: OpenAICompatibleLLM, context: dict[str, Any]) -> dict[str, Any]:
    content = llm.chat(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        json_mode=True,
        temperature=0.0,
    )
    plan = parse_json_object(content)
    if "nodes" not in plan:
        raise AgentSystemError("Planner JSON must contain a top-level 'nodes' array")
    return plan


def build_execution_graph(plan: dict[str, Any]) -> dict[str, Any]:
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise AgentSystemError("Plan must contain a non-empty nodes list")

    by_id: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise AgentSystemError(f"Plan node {idx} must be an object")
        node_id = str(node.get("id") or f"node_{idx + 1}")
        if node_id in by_id:
            raise AgentSystemError(f"Duplicate node id: {node_id}")
        agent = node.get("agent")
        if agent not in AGENT_CALLERS:
            raise AgentSystemError(f"Unsupported agent {agent!r} in node {node_id}")
        node = {**node, "id": node_id, "depends_on": [str(dep) for dep in node.get("depends_on", [])]}
        by_id[node_id] = node

    for node in by_id.values():
        for dep in node["depends_on"]:
            if dep not in by_id:
                raise AgentSystemError(f"Node {node['id']} depends on unknown node {dep}")

    ordered: list[list[str]] = []
    pending = set(by_id)
    completed: set[str] = set()
    while pending:
        ready = sorted(node_id for node_id in pending if set(by_id[node_id]["depends_on"]) <= completed)
        if not ready:
            raise AgentSystemError("Plan contains a dependency cycle")
        ordered.append(ready)
        completed.update(ready)
        pending.difference_update(ready)

    return {"nodes": list(by_id.values()), "levels": ordered}


def execute_graph(graph: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    results: dict[str, Any] = {}

    for level in graph["levels"]:
        if len(level) == 1:
            node_id = level[0]
            results[node_id] = execute_node(nodes_by_id[node_id], context, results)
            continue
        with ThreadPoolExecutor(max_workers=len(level)) as pool:
            futures = {
                pool.submit(execute_node, nodes_by_id[node_id], context, results): node_id
                for node_id in level
            }
            for future in as_completed(futures):
                node_id = futures[future]
                results[node_id] = future.result()
    return results


def execute_node(node: dict[str, Any], context: dict[str, Any], prior_results: dict[str, Any]) -> dict[str, Any]:
    agent = str(node["agent"])
    params = resolve_params(node.get("params") or {}, context, prior_results)
    params = apply_defaults(agent, params, context, node)
    try:
        return AGENT_CALLERS[agent](**params)
    except Exception as exc:
        return {
            "model": agent,
            "task": params.get("task"),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "params": params,
        }


def resolve_params(params: Any, context: dict[str, Any], prior_results: dict[str, Any]) -> Any:
    if isinstance(params, dict):
        return {key: resolve_params(value, context, prior_results) for key, value in params.items()}
    if isinstance(params, list):
        return [resolve_params(item, context, prior_results) for item in params]
    if isinstance(params, str):
        match = re.fullmatch(r"\$\{([^}]+)\}", params)
        if match:
            return lookup_reference(match.group(1), context, prior_results)
    return params


def lookup_reference(ref: str, context: dict[str, Any], prior_results: dict[str, Any]) -> Any:
    root_name, _, tail = ref.partition(".")
    if root_name == "context":
        value: Any = context
    elif root_name in prior_results:
        value = prior_results[root_name]
    else:
        raise AgentSystemError(f"Unknown parameter reference: {ref}")
    for part in tail.split(".") if tail else []:
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list):
            value = value[int(part)]
        else:
            raise AgentSystemError(f"Cannot resolve {ref!r} through non-container value")
    return value


def apply_defaults(agent: str, params: dict[str, Any], context: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    output_root = Path(context["output_dir"]) / str(node["id"])
    params.setdefault("output_dir", str(output_root))

    if agent == "skyeyegpt":
        params.setdefault("image_path", context.get("image_path"))
        params.setdefault("prompt", context["user_query"])
        params.setdefault("history", context.get("history") or [])
        params.setdefault("task", "vqa" if context.get("image_path") else "chat")
    elif agent == "sarmae":
        params.setdefault("image_path", context.get("image_path"))
        params.setdefault("task", "detect")
    elif agent == "mtp":
        params.setdefault("image_path", context.get("image_path"))
        params.setdefault("task", "rotated")
    elif agent == "dofa":
        params.pop("task", None)
        if context.get("dataset") is not None:
            params.setdefault("dataset", context["dataset"])
        if context.get("sample_id") is not None:
            params.setdefault("sample_id", context["sample_id"])
        if context.get("input_path") is not None:
            params.setdefault("input_path", context["input_path"])
        input_path = params.get("input_path")
        if input_path and Path(str(input_path)).suffix.lower() not in {
            ".hdf5",
            ".h5",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
        }:
            params.pop("input_path", None)
    elif agent == "sattxt":
        params.setdefault("image_paths", context.get("image_paths") or [])
        params.setdefault("texts", context.get("texts") or [])
        params.setdefault("categories", context.get("categories") or [])
    elif agent == "dehazeformer":
        params.setdefault("image_path", context.get("image_path"))

    missing = required_missing(agent, params)
    if missing:
        raise AgentSystemError(f"Node {node['id']} ({agent}) is missing required params: {', '.join(missing)}")
    return params


def required_missing(agent: str, params: dict[str, Any]) -> list[str]:
    required = {
        "skyeyegpt": [],
        "sarmae": ["image_path"],
        "mtp": ["image_path"],
        "dofa": ["dataset"],
        "sattxt": ["task"],
        "dehazeformer": ["image_path"],
    }[agent]
    missing = []
    for key in required:
        value = params.get(key)
        if value is None or value == "" or value == []:
            missing.append(key)
    return missing


def synthesize_answer(
    llm: OpenAICompatibleLLM,
    context: dict[str, Any],
    plan: dict[str, Any],
    agent_results: dict[str, Any],
) -> str:
    compact_results = compact_for_llm(agent_results)
    content = llm.chat(
        [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"context": context, "plan": plan, "agent_results": compact_results},
                    ensure_ascii=False,
                ),
            },
        ],
        json_mode=False,
        temperature=0.2,
    )
    return str(content).strip()


def synthesize_answer_without_llm(
    context: dict[str, Any],
    plan: dict[str, Any],
    agent_results: dict[str, Any],
) -> str:
    """Deterministic fallback used when a manual plan is executed without API keys."""

    if not agent_results:
        return "任务已规划，但还没有执行任何 Agent。"

    lines = []
    for node in plan.get("nodes", []):
        node_id = str(node.get("id"))
        result = agent_results.get(node_id)
        if not isinstance(result, dict):
            lines.append(f"{node_id}: 已返回结果。")
            continue

        if result.get("status") == "unavailable":
            lines.append(f"{node_id}: {result.get('model')} 暂不可用，原因是 {result.get('reason')}")
            continue
        if result.get("status") == "error":
            lines.append(f"{node_id}: {result.get('model')} 调用失败，错误是 {result.get('error_type')}: {result.get('error')}")
            continue

        model = result.get("model", node.get("agent"))
        task = result.get("task", node.get("params", {}).get("task"))
        if model == "SARMAE" and task == "detect":
            lines.append(f"图中检测到 {int(result.get('num_detections', 0))} 个目标。")
        elif model == "SARMAE" and task == "segment":
            lines.append(f"SAR 分割已完成，结果图: {result.get('visualization') or result.get('mask')}")
        elif model == "MTP" and task in {"horizontal", "rotated"}:
            box_name = "水平框" if task == "horizontal" else "旋转框"
            target = result.get("target_class")
            target_text = f"{target} 类目标" if target else "目标"
            lines.append(f"MTP {box_name}检测完成，共检测到 {int(result.get('num_detections', 0))} 个{target_text}，结果图: {result.get('visualization')}")
        elif model == "MTP" and task == "both":
            h_count = int((result.get("horizontal") or {}).get("num_detections", 0))
            r_count = int((result.get("rotated") or {}).get("num_detections", 0))
            lines.append(f"MTP 检测完成，水平框 {h_count} 个，旋转框 {r_count} 个。")
        elif model == "DOFA":
            distribution = format_class_distribution(result.get("class_distribution"))
            message = f"遥感语义分割已完成，预测图: {result.get('prediction')}，叠加图: {result.get('overlay')}"
            if distribution:
                message += f"。类别占比: {distribution}"
            lines.append(message)
        elif model == "DehazeFormer":
            lines.append(f"云雾/雾霾去除已完成，输出图像: {result.get('output_image')}")
        elif model == "SATtxt" and result.get("prediction"):
            lines.append(f"零样本分类结果是: {result.get('prediction')}。")
        elif model == "SATtxt" and result.get("image_to_text"):
            best = best_image_to_text_match(result)
            if best:
                lines.append(f"最匹配的文本是: {best['text']}，相似度分数为 {best['score']:.4f}。")
            else:
                lines.append(f"{node_id}: SATtxt 已完成图文检索。")
        elif model == "SkyEyeGPT" and result.get("answer"):
            lines.append(str(result["answer"]))
        else:
            lines.append(f"{node_id}: {model} 已完成 {task}。")

    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines)


def best_image_to_text_match(result: dict[str, Any]) -> dict[str, Any] | None:
    image_to_text = result.get("image_to_text")
    if not isinstance(image_to_text, list) or not image_to_text:
        return None
    first = image_to_text[0]
    if not isinstance(first, dict):
        return None
    ranking = first.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return None
    top = ranking[0]
    if not isinstance(top, dict) or "text" not in top:
        return None
    return {"text": str(top["text"]), "score": float(top.get("score", 0.0))}


def format_class_distribution(distribution: Any) -> str:
    if not isinstance(distribution, list):
        return ""
    parts = []
    for item in distribution:
        if not isinstance(item, dict):
            continue
        name = item.get("class_name", f"class_{item.get('class_id')}")
        ratio = float(item.get("ratio", 0.0)) * 100.0
        parts.append(f"{name} {ratio:.2f}%")
    return "，".join(parts)


def compact_for_llm(value: Any, max_items: int = 30) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {"similarity"}:
                continue
            out[key] = compact_for_llm(item, max_items=max_items)
        return out
    if isinstance(value, list):
        return [compact_for_llm(item, max_items=max_items) for item in value[:max_items]]
    return value


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise AgentSystemError("Expected a JSON object")
    return value


AGENT_CALLERS = {
    "skyeyegpt": skyeyegpt,
    "sarmae": sarmae,
    "mtp": mtp,
    "dofa": dofa,
    "dehazeformer": dehazeformer,
    "sattxt": sattxt,
}


PLANNER_SYSTEM_PROMPT = """你是卫星遥感多模型系统的任务规划器。只输出 JSON 对象，不要输出解释。

可用 Agent:
1. skyeyegpt: 图像描述、VQA、视觉对话、开放式空间关系判断、需要自然语言理解的图像问题。
   params: image_path, prompt, task(chat|caption|vqa|grounding), history, max_new_tokens, temperature。
2. sarmae: SAR 图像检测或 SAR 语义分割。
   params: image_path, task(detect|segment|both), score_thr。
3. mtp: 光学遥感目标检测和目标计数，支持水平框和旋转框；可按 DIOR 类别过滤。
   params: image_path, task(horizontal|hbb|rotated|obb|both), score_thr, target_class。
   target_class 可用 DIOR 英文类名：airplane, airport, baseballfield, basketballcourt, bridge, chimney, expressway-service-area, expressway-toll-station, dam, golffield, groundtrackfield, harbor, overpass, ship, stadium, storagetank, tenniscourt, trainstation, vehicle, windmill。
4. dofa: 遥感语义分割。
   params: dataset, sample_id, input_path, split, crop_size, stride, batch_size。
5. dehazeformer: 单张 RGB 云雾/雾霾图像去云、去雾、图像复原。
   params: image_path, model_name, exp, checkpoint, device。
6. sattxt: 零样本分类、图文检索、以文本类别或文本候选项匹配图像。
   params: task(zero_shot_classification|retrieval|image_to_text|text_to_image), image_paths, texts, categories。

输出 JSON schema:
{
  "task_type": "detection|segmentation|counting|spatial_relation|caption|vqa|chat|dehazing|restoration|zero_shot_classification|retrieval|composite",
  "reasoning": "一句话说明为什么这样拆解",
  "nodes": [
    {
      "id": "短英文标识",
      "agent": "skyeyegpt|sarmae|mtp|dofa|dehazeformer|sattxt",
      "depends_on": [],
      "params": {}
    }
  ]
}

规划规则:
- 用户要求图像描述、VQA、对话，使用 skyeyegpt。
- SAR 检测或 SAR 分割，使用 sarmae；只有在用户问题或用户显式提供的上下文明确说明这是 SAR/合成孔径雷达图像时，才按 SAR 图像处理并使用 sarmae，不要根据文件路径或数据集目录名推断图像类型。
- 光学遥感目标检测、DIOR 水平框检测使用 mtp task=horizontal；旋转框/方向框检测使用 mtp task=rotated；如果用户同时要求两种框，使用 mtp task=both。
- 用户问“检测图像中的 xx”“图像中有几个 xx”“数一数 xx”等目标检测/计数问题时，先判断图像类型：SAR 图像使用 sarmae；光学遥感图像才使用 mtp。如果能从光学问题中识别目标类别，把 target_class 设为对应 DIOR 英文类名，例如 船/船只=ship，车辆/车=vehicle，飞机=airplane，桥/桥梁=bridge，风车=windmill，储油罐/油罐=storagetank，篮球场=basketballcourt，网球场=tenniscourt。
- 遥感语义分割，使用 dofa；如果用户没有给 dataset，不要编造 dataset，保留空 params 让系统报缺参。DOFA 的 input_path 可以是 HDF5；RGB-only head 也可接收普通 jpg/png/tif 图像。
- 用户要求去云、去雾、去 haze、dehaze、remove cloud/fog/haze 或云雾图像复原，使用 dehazeformer。
- 零样本分类或检索，使用 sattxt。
- 计数如果是 SAR 目标计数，先用 sarmae detect；如果是光学遥感目标计数，先用 mtp detect，并在可识别类别时设置 target_class；最终回答阶段根据 num_detections 作答。
- 开放式空间关系判断优先 skyeyegpt；如果明确需要 bbox 中间结果，可以先 detect，再用 skyeyegpt 或最终回答整合。
- 复合任务要把中间结果拆成节点；能并行的节点使用相同 depends_on，串行节点用 depends_on 引用上一步。
- 参数可以用 ${context.image_path}, ${context.image_paths}, ${context.categories}, ${context.texts}, ${context.dataset}, ${context.sample_id}, ${context.input_path}, ${node_id.some.field} 引用。
"""


SYNTHESIZER_SYSTEM_PROMPT = """你是卫星遥感多模型系统的最终回答器。根据用户问题、执行计划和 Agent 结果，用中文给出简洁、可信的人类可读回答。

要求:
- 如果 Agent 已经给出自然语言答案，做轻微润色即可。
- 如果结果是 bbox/detection，优先用 num_detections 回答数量，并可说明置信度最高目标。
- 如果结果是 segmentation，说明已完成分割，并给出 mask/visualization/prediction 路径和可解释的统计信息。
- 如果结果是零样本分类，回答 prediction，并可概括主要 scores。
- 如果某个 Agent 返回 status=unavailable 或错误信息，要明确说明未完成的原因，不要假装成功。
- 不要输出 JSON。
"""
