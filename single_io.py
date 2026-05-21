"""Single JSON input/output entry point for SmartQG.

Examples:
  # Generate one question from an inline KG
  python single_io.py --mode generate --input examples/generate_request.json --output outputs/question_output.json

  # Generate one question by referencing a saved KG file
  python single_io.py --mode generate --input examples/generate_with_kg_path_request.json --output outputs/question_output.json

  # Build one KG from raw text and save it into the KG store
  python single_io.py --mode build_kg --input examples/build_kg_request.json --output kg_store/binary_search.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from rag_system.json_io import DEFAULT_KG_STORE_DIR, generate_question_from_json

# Force the local .env file to take precedence over stale shell variables.
load_dotenv(".env", override=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object/dict.")

    # Internal helper used by json_io.py for resolving relative kg_path values.
    data.setdefault("_request_dir", str(Path(path).expanduser().resolve().parent))
    return data


def _write_json(data: Dict[str, Any], path: Optional[str]) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
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
      "topic": "merge sort",
      "kg_id": "merge_sort",              # optional, useful when saving into kg_store/
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
        "kg_id": request.get("kg_id", ""),
        "knowledge_graph": kg,
        "metadata": {
            "mode": "single_kg_build",
            "input_texts": len(texts),
            "kg_nodes": len(kg.get("nodes", [])),
            "kg_relations": len(kg.get("relations", [])),
            "kg_triplets": len(kg.get("triplets", [])),
        },
    }


def _default_build_kg_output(request: Dict[str, Any], kg_store_dir: str) -> Optional[str]:
    """Return kg_store/<kg_id>.json when build mode has kg_id but no --output."""
    kg_id = str(request.get("kg_id") or "").strip()
    if not kg_id:
        return None
    filename = kg_id if kg_id.endswith(".json") else f"{kg_id}.json"
    return str(Path(kg_store_dir) / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartQG single JSON input/output runner")
    parser.add_argument("--mode", choices=["generate", "build_kg"], default="generate")
    parser.add_argument("--input", required=True, help="Path to JSON request")
    parser.add_argument("--output", default=None, help="Optional path to write JSON response")
    parser.add_argument(
        "--kg-store",
        default=os.getenv("SMARTQG_KG_STORE", DEFAULT_KG_STORE_DIR),
        help="Default KG store directory used when build_kg has kg_id but --output is omitted.",
    )
    args = parser.parse_args()

    request = _read_json(args.input)
    if args.mode == "build_kg":
        response = build_kg_from_json(request)
        output_path = args.output or _default_build_kg_output(request, args.kg_store)
    else:
        response = generate_question_from_json(request)
        output_path = args.output
    _write_json(response, output_path)


if __name__ == "__main__":
    main()
