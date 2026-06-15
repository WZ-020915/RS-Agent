from __future__ import annotations

import cgi
import html
import json
import mimetypes
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_wrappers import run_remote_sensing_agent_system  # noqa: E402
from model_wrappers.agent_system import best_image_to_text_match, format_class_distribution, synthesize_answer_without_llm  # noqa: E402


HOST = "0.0.0.0"
PORT = 7860
UPLOAD_DIR = Path("/root/autodl-tmp/model_wrappers/outputs/agent_system/uploads")
SAFE_FILE_ROOT = Path("/root/autodl-tmp").resolve()
DEFAULT_FORM_STATE = {
    "query": "图中有几个船？",
    "image_path": "/root/autodl-tmp/SARMAE-main/Official-SSDD-OPEN/BBox_SSDD/coco_style/images/test/000001.jpg",
    "image_paths": "",
    "categories": "",
    "texts": "",
    "dataset": "",
    "sample_id": "",
    "input_path": "",
    "execute": True,
    "no_llm_final": True,
}


def parse_json_list(value: str) -> list[str] | None:
    value = value.strip()
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Expected a JSON string list")
    return parsed


def summarize_result(result: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return result
    plan = result.get("plan") or {}
    nodes = plan.get("nodes") or []
    summary = {
        "status": result.get("status"),
        "answer": result.get("answer") or ("仅完成任务规划；勾选“执行 Agent”才会调用模型。" if result.get("status") == "planned" else None),
        "task_type": plan.get("task_type"),
        "reasoning": plan.get("reasoning"),
        "agents": [
            {
                "id": node.get("id"),
                "agent": node.get("agent"),
                "task": (node.get("params") or {}).get("task"),
            }
            for node in nodes
        ],
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
        item = {"model": value.get("model"), "task": value.get("task"), "status": value.get("status", "ok")}
        for key in [
            "answer",
            "prediction",
            "num_detections",
            "visualization",
            "mask",
            "raw_prediction",
            "overlay",
            "rgb",
            "compare",
            "class_names",
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


def collect_output_files(value: object) -> list[str]:
    keys = {"visualization", "prediction", "overlay", "rgb", "compare", "mask", "raw_prediction"}
    files = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str) and Path(item).exists():
                files.append(item)
            else:
                files.extend(collect_output_files(item))
    elif isinstance(value, list):
        for item in value:
            files.extend(collect_output_files(item))
    return sorted(set(files))


def file_url(path: str) -> str:
    return "/file?path=" + quote(path)


def field_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    item = form[name] if name in form else None
    if item is None or item.filename:
        return default
    return str(item.value)


def save_upload(form: cgi.FieldStorage) -> str | None:
    if "image_file" not in form:
        return None
    item = form["image_file"]
    if not item.filename:
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(item.filename).name
    path = UPLOAD_DIR / safe_name
    counter = 1
    while path.exists():
        path = UPLOAD_DIR / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
        counter += 1
    data = item.file.read()
    path.write_bytes(data)
    return str(path)


def state_from_multipart(form: cgi.FieldStorage, uploaded: str | None = None) -> dict:
    state = {
        "query": field_value(form, "query", DEFAULT_FORM_STATE["query"]),
        "image_path": uploaded or field_value(form, "image_path"),
        "image_paths": field_value(form, "image_paths"),
        "categories": field_value(form, "categories"),
        "texts": field_value(form, "texts"),
        "dataset": field_value(form, "dataset"),
        "sample_id": field_value(form, "sample_id"),
        "input_path": field_value(form, "input_path"),
        "execute": field_value(form, "execute") == "on",
        "no_llm_final": field_value(form, "no_llm_final") == "on",
    }
    return state


def state_from_urlencoded(data: dict[str, list[str]]) -> dict:
    get = lambda name, default="": data.get(name, [default])[0]
    return {
        "query": get("query", DEFAULT_FORM_STATE["query"]),
        "image_path": get("image_path"),
        "image_paths": get("image_paths"),
        "categories": get("categories"),
        "texts": get("texts"),
        "dataset": get("dataset"),
        "sample_id": get("sample_id"),
        "input_path": get("input_path"),
        "execute": get("execute") == "on",
        "no_llm_final": get("no_llm_final") == "on",
    }


def clean_state(state: dict) -> dict:
    cleaned = dict(DEFAULT_FORM_STATE)
    cleaned.update(state)
    for key in ["query", "image_path", "image_paths", "categories", "texts", "dataset", "sample_id", "input_path"]:
        cleaned[key] = str(cleaned.get(key) or "").strip()
    cleaned["execute"] = bool(cleaned.get("execute"))
    cleaned["no_llm_final"] = bool(cleaned.get("no_llm_final"))
    return cleaned


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/file":
            self.respond_file(parsed)
            return
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        self.respond_html(render_page())

    def do_POST(self) -> None:
        state = dict(DEFAULT_FORM_STATE)
        try:
            content_type = self.headers.get("Content-Type", "")
            if content_type.startswith("multipart/form-data"):
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
                uploaded = save_upload(form)
                state = clean_state(state_from_multipart(form, uploaded))
                result = self.run_from_state(state)
            else:
                length = int(self.headers.get("Content-Length", "0"))
                data = parse_qs(self.rfile.read(length).decode("utf-8"))
                state = clean_state(state_from_urlencoded(data))
                result = self.run_from_state(state)
            self.respond_html(render_page(result=result, form_state=state))
        except Exception as exc:
            payload = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            self.respond_html(render_page(result=payload, form_state=state), status=500)

    def run_from_state(self, state: dict) -> dict:
        return run_remote_sensing_agent_system(
            state["query"],
            image_path=state["image_path"] or None,
            image_paths=parse_json_list(state["image_paths"]),
            categories=parse_json_list(state["categories"]),
            texts=parse_json_list(state["texts"]),
            dataset=state["dataset"] or None,
            sample_id=state["sample_id"] or None,
            input_path=state["input_path"] or None,
            execute=state["execute"],
            synthesizer=synthesize_answer_without_llm if state["no_llm_final"] else None,
        )

    def respond_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_file(self, parsed) -> None:
        query = parse_qs(parsed.query)
        raw_path = query.get("path", [""])[0]
        path = Path(raw_path).resolve()
        if not str(path).startswith(str(SAFE_FILE_ROOT)) or not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def render_page(result: dict | None = None, form_state: dict | None = None) -> str:
    form_state = clean_state(form_state or DEFAULT_FORM_STATE)
    summary = summarize_result(result)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2) if summary is not None else ""
    result_json = json.dumps(result, ensure_ascii=False, indent=2) if result is not None else ""
    answer = ""
    if isinstance(summary, dict):
        answer = summary.get("answer") or summary.get("status", "")
    output_files = collect_output_files(summary)
    output_links = "\n".join(
        f'<li><a href="{html.escape(file_url(path))}" target="_blank">{html.escape(Path(path).name)}</a>'
        f'<div class="path">{html.escape(path)}</div></li>'
        for path in output_files
    )
    checked_execute = " checked" if form_state["execute"] else ""
    checked_no_llm_final = " checked" if form_state["no_llm_final"] else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Satellite Agent Demo</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f4; color: #1d2528; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 24px; margin: 0 0 20px; }}
    form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    label {{ display: grid; gap: 6px; font-size: 14px; font-weight: 600; }}
    textarea, input {{ box-sizing: border-box; width: 100%; border: 1px solid #b9c0bd; border-radius: 6px; padding: 10px; font-size: 14px; background: white; }}
    textarea {{ min-height: 96px; resize: vertical; }}
    .wide {{ grid-column: 1 / -1; }}
    .checks {{ display: flex; gap: 18px; align-items: center; grid-column: 1 / -1; }}
    .checks label {{ display: flex; grid-template-columns: none; align-items: center; gap: 8px; font-weight: 600; }}
    .checks input {{ width: auto; }}
    button {{ width: 160px; border: 0; border-radius: 6px; padding: 11px 14px; background: #135e57; color: white; font-weight: 700; cursor: pointer; }}
    section {{ margin-top: 24px; }}
    ul {{ padding-left: 20px; }}
    a {{ color: #135e57; font-weight: 700; }}
    .path {{ color: #596360; font-size: 12px; margin: 4px 0 10px; word-break: break-all; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #101617; color: #d8eee7; padding: 16px; border-radius: 6px; overflow: auto; }}
    details {{ margin-top: 18px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .answer {{ background: white; border: 1px solid #d5d9d6; border-radius: 6px; padding: 14px; min-height: 24px; }}
  </style>
</head>
<body>
<main>
  <h1>Satellite Multi-Agent Demo</h1>
  <form method="post" enctype="multipart/form-data">
    <label class="wide">用户问题
      <textarea name="query">{html.escape(form_state["query"])}</textarea>
    </label>
    <label>上传图片
      <input type="file" name="image_file" accept="image/*">
    </label>
    <label>或填写本地图片路径
      <input name="image_path" value="{html.escape(form_state["image_path"])}">
    </label>
    <label>多图路径 JSON
      <input name="image_paths" value="{html.escape(form_state["image_paths"])}" placeholder='["/path/a.jpg", "/path/b.jpg"]'>
    </label>
    <label>零样本类别 JSON
      <input name="categories" value="{html.escape(form_state["categories"])}" placeholder='["port", "forest", "airport"]'>
    </label>
    <label>检索文本 JSON
      <input name="texts" value="{html.escape(form_state["texts"])}" placeholder='["a port satellite image", "a forest satellite image"]'>
    </label>
    <label>DOFA dataset
      <input name="dataset" value="{html.escape(form_state["dataset"])}" placeholder="m-pv4ger-seg">
    </label>
    <label>DOFA sample_id
      <input name="sample_id" value="{html.escape(form_state["sample_id"])}">
    </label>
    <label>DOFA input_path
      <input name="input_path" value="{html.escape(form_state["input_path"])}">
    </label>
    <div class="checks">
      <label><input type="checkbox" name="execute"{checked_execute}> 执行 Agent</label>
      <label><input type="checkbox" name="no_llm_final"{checked_no_llm_final}> 执行后不用第二次 LLM 汇总</label>
    </div>
    <button type="submit">运行</button>
  </form>
  <section>
    <h2>回答</h2>
    <div class="answer">{html.escape(str(answer))}</div>
  </section>
  <section>
    <h2>输出文件</h2>
    <ul>{output_links or "<li>暂无输出文件</li>"}</ul>
  </section>
  <section>
    <h2>摘要 JSON</h2>
    <pre>{html.escape(summary_json)}</pre>
    <details>
      <summary>完整 plan / graph / raw results</summary>
      <pre>{html.escape(result_json)}</pre>
    </details>
  </section>
</main>
</body>
</html>"""


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
