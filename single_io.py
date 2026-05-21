"""Single JSON input/output entry point for SmartQG.

Examples:
  python single_io.py --mode generate --input examples/generate_request.json --output output.json
  python single_io.py --mode build_kg --input examples/build_kg_request.json --output kg.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from rag_system.json_io import generate_question_from_json

from dotenv import load_dotenv
load_dotenv(".env", override=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object/dict.")
    return data


def _write_json(data: Dict[str, Any], path: Optional[str]) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


def _normalize_text_items(request: Dict[str, Any]) -> List[Dict[str, str]]:
    """Accept either texts: [..] or a single text field for one-shot KG building."""
    raw_texts = request.get("texts")
    if raw_texts is None and request.get("text"):
        raw_texts = [{"node_id": "input_node_0", "content": request["text"]}]
    if not isinstance(raw_texts, list) or not raw_texts:
        raise ValueError("build_kg mode requires `texts` list or a single `text` string.")

    items: List[Dict[str, str]] = []
    for i, item in enumerate(raw_texts):
        if isinstance(item, str):
            content = item.strip()
            node_id = f"input_node_{i}"
        elif isinstance(item, dict):
            content = str(item.get("content") or item.get("text") or "").strip()
            node_id = str(item.get("node_id") or item.get("id") or f"input_node_{i}")
        else:
            continue
        if content:
            items.append({"node_id": node_id, "content": content})
    if not items:
        raise ValueError("No non-empty text found for KG construction.")
    return items


def build_kg_from_json(request: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Build a KG JSON from one JSON request, independent of question generation.

    Request schema:
    {
      "topic": "merge sort",              # optional but recommended
      "texts": [
        {"node_id": "n1", "content": "..."},
        {"node_id": "n2", "content": "..."}
      ]
    }
    """
    resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_api_key:
        raise ValueError("DEEPSEEK_API_KEY is required, either in environment or as api_key.")

    from build_kg import LLMGraphExtractor

    texts = _normalize_text_items(request)
    extractor = LLMGraphExtractor(resolved_api_key, output_file=None)
    kg = extractor.build_from_texts(texts)
    return {
        "topic": request.get("topic", ""),
        "knowledge_graph": kg,
        "metadata": {
            "mode": "single_kg_build",
            "input_texts": len(texts),
            "kg_nodes": len(kg.get("nodes", [])),
            "kg_relations": len(kg.get("relations", [])),
            "kg_triplets": len(kg.get("triplets", [])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartQG single JSON input/output runner")
    parser.add_argument("--mode", choices=["generate", "build_kg"], default="generate")
    parser.add_argument("--input", required=True, help="Path to JSON request")
    parser.add_argument("--output", default=None, help="Optional path to write JSON response")
    args = parser.parse_args()

    request = _read_json(args.input)
    if args.mode == "build_kg":
        response = build_kg_from_json(request)
    else:
        response = generate_question_from_json(request)
    _write_json(response, args.output)


if __name__ == "__main__":
    main()
