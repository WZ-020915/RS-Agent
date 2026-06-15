from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_wrappers import run_remote_sensing_agent_system  # noqa: E402
from model_wrappers.agent_system import synthesize_answer_without_llm  # noqa: E402


def parse_json_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a JSON string list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the remote-sensing multi-agent system.")
    parser.add_argument("query", help="User query in natural language")
    parser.add_argument("--image-path")
    parser.add_argument("--image-paths", help='JSON list, e.g. ["a.jpg", "b.jpg"]')
    parser.add_argument("--categories", help='JSON list for zero-shot classification, e.g. ["port", "forest"]')
    parser.add_argument("--texts", help='JSON list for retrieval candidates or text queries')
    parser.add_argument("--dataset", help="DOFA dataset/head name")
    parser.add_argument("--sample-id")
    parser.add_argument("--input-path")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/model_wrappers/outputs/agent_system")
    parser.add_argument("--planner-json", help="Path to a saved planner JSON file")
    parser.add_argument("--plan-only", action="store_true", help="Only call/validate the planner; do not run model agents")
    parser.add_argument(
        "--no-llm-final",
        action="store_true",
        help="With --planner-json, use the deterministic final summarizer instead of calling an LLM",
    )
    args = parser.parse_args()

    planner_json = None
    if args.planner_json:
        planner_json = json.loads(Path(args.planner_json).read_text())

    result = run_remote_sensing_agent_system(
        args.query,
        image_path=args.image_path,
        image_paths=parse_json_list(args.image_paths),
        categories=parse_json_list(args.categories),
        texts=parse_json_list(args.texts),
        dataset=args.dataset,
        sample_id=args.sample_id,
        input_path=args.input_path,
        output_dir=args.output_dir,
        planner_json=planner_json,
        synthesizer=synthesize_answer_without_llm if args.no_llm_final else None,
        execute=not args.plan_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
