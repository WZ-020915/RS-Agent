from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_wrappers import run_remote_sensing_agent_system  # noqa: E402
from model_wrappers.agent_system import (  # noqa: E402
    best_image_to_text_match,
    format_class_distribution,
    synthesize_answer_without_llm,
)


def parse_json_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Expected a JSON string list")
    return parsed


def summarize_result(result: dict) -> dict:
    plan = result.get("plan") or {}
    nodes = plan.get("nodes") or []
    agents = [
        {
            "id": node.get("id"),
            "agent": node.get("agent"),
            "task": (node.get("params") or {}).get("task"),
        }
        for node in nodes
    ]
    summary = {
        "status": result.get("status"),
        "answer": result.get("answer") or ("仅完成任务规划；加 --execute 才会调用 Agent。" if result.get("status") == "planned" else None),
        "task_type": plan.get("task_type"),
        "reasoning": plan.get("reasoning"),
        "agents": agents,
    }
    if "agent_results" in result:
        summary["agent_results"] = summarize_agent_results(result["agent_results"])
    return summary


def summarize_agent_results(agent_results: dict) -> dict:
    out = {}
    for node_id, value in agent_results.items():
        if not isinstance(value, dict):
            out[node_id] = value
            continue
        item = {
            "model": value.get("model"),
            "task": value.get("task"),
            "status": value.get("status", "ok"),
        }
        for key in [
            "answer",
            "prediction",
            "num_detections",
            "visualization",
            "mask",
            "raw_prediction",
            "prediction",
            "overlay",
            "rgb",
            "compare",
            "class_names",
            "target_class",
            "class_distribution",
            "horizontal",
            "rotated",
            "error_type",
            "error",
        ]:
            if key in value:
                item[key] = value[key]
        distribution = format_class_distribution(value.get("class_distribution"))
        if distribution:
            item["class_summary"] = distribution
        best_text = best_image_to_text_match(value)
        if best_text:
            item["best_text"] = best_text["text"]
            item["best_score"] = best_text["score"]
        out[node_id] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the full remote-sensing agent chain.")
    parser.add_argument("--query", default="图中有几个船？")
    parser.add_argument("--image-path")
    parser.add_argument("--image-paths", help='JSON list, e.g. ["a.jpg", "b.jpg"]')
    parser.add_argument("--categories", help='JSON list for SATtxt zero-shot classification')
    parser.add_argument("--texts", help="JSON list for SATtxt retrieval")
    parser.add_argument("--dataset", help="DOFA dataset/head name")
    parser.add_argument("--sample-id")
    parser.add_argument("--input-path")
    parser.add_argument("--execute", action="store_true", help="Actually run selected model agents")
    parser.add_argument(
        "--no-llm-final",
        action="store_true",
        help="Use deterministic final summary after agent execution instead of a second LLM call",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full plan, graph, and raw agent results")
    args = parser.parse_args()

    result = run_remote_sensing_agent_system(
        args.query,
        image_path=args.image_path,
        image_paths=parse_json_list(args.image_paths),
        categories=parse_json_list(args.categories),
        texts=parse_json_list(args.texts),
        dataset=args.dataset,
        sample_id=args.sample_id,
        input_path=args.input_path,
        execute=args.execute,
        synthesizer=synthesize_answer_without_llm if args.no_llm_final else None,
    )
    output = result if args.verbose else summarize_result(result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
