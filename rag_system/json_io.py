"""JSON input/output helpers for single-question generation.

This module converts a user-provided KG JSON into the ``graph_context`` format
expected by ``SmartGenerator`` and wraps the raw generator output into one stable
JSON response object.

Supported KG inputs for question generation:
1. Inline KG: ``{"knowledge_graph": {...}}``
2. KG file path: ``{"kg_path": "kg_store/binary_search.json"}``
3. KG registry id: ``{"kg_id": "binary_search"}``, resolved to
   ``kg_store/binary_search.json`` by default.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_QUESTION_FORMATS = {
    "mcq_single",
    "mcq_multi",
    "true_false",
    "fill_blank",
    "open_answer",
}

VALID_QUESTION_TYPES = {"computational", "conceptual"}
DEFAULT_KG_STORE_DIR = "kg_store"


def _as_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object/dict.")
    return value


def _request_dir(req: Dict[str, Any]) -> Optional[Path]:
    """Return the directory of the request JSON when called from ``single_io.py``.

    ``single_io.py`` injects ``_request_dir`` internally so that relative
    ``kg_path`` values can be resolved relative to the input JSON file. This key
    is internal and is never required from users.
    """
    raw = req.get("_request_dir")
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser().resolve()
    except Exception:
        return None


def _candidate_relative_paths(raw_path: str, req: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Return possible filesystem locations for a user-provided relative path."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return [path]

    candidates: List[Path] = []
    if req is not None:
        rdir = _request_dir(req)
        if rdir is not None:
            candidates.append(rdir / path)

    # Most users run commands from the project root, so keep CWD as a fallback.
    candidates.append(Path.cwd() / path)
    candidates.append(path)

    # Preserve order while removing duplicates.
    seen = set()
    unique: List[Path] = []
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def _first_existing_path(candidates: List[Path]) -> Path:
    for cand in candidates:
        if cand.exists():
            return cand
    # Return the first candidate so error messages remain actionable.
    return candidates[0]


def load_knowledge_graph_from_file(kg_path: str, request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a KG JSON file.

    The file may either contain the KG directly:

    ``{"nodes": [...], "relations": [...]}``

    or a wrapper produced by build mode:

    ``{"topic": "...", "knowledge_graph": {"nodes": [...], ...}}``
    """
    candidates = _candidate_relative_paths(kg_path, request)
    path = _first_existing_path(candidates)
    if not path.exists():
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Knowledge graph file not found. Tried: {tried}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"KG file must contain a JSON object/dict: {path}")

    if isinstance(data.get("knowledge_graph"), dict):
        return data["knowledge_graph"]
    if isinstance(data.get("kg"), dict):
        return data["kg"]
    if isinstance(data.get("graph_context"), dict):
        return data["graph_context"]
    return data


def _kg_store_candidates(req: Dict[str, Any]) -> List[Path]:
    raw_store_dir = str(req.get("kg_store_dir") or os.getenv("SMARTQG_KG_STORE") or DEFAULT_KG_STORE_DIR)
    path = Path(raw_store_dir).expanduser()
    if path.is_absolute():
        return [path]

    candidates: List[Path] = [Path.cwd() / path]
    rdir = _request_dir(req)
    if rdir is not None:
        candidates.append(rdir / path)

    # Preserve order while removing duplicates.
    seen = set()
    unique: List[Path] = []
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def resolve_kg_path_from_id(kg_id: str, request: Dict[str, Any]) -> Path:
    """Resolve ``kg_id`` to a KG JSON file path.

    ``kg_id`` may be provided either with or without ``.json``. By default,
    ``binary_search`` resolves to ``kg_store/binary_search.json``.
    """
    kg_id = str(kg_id).strip()
    if not kg_id:
        raise ValueError("kg_id cannot be empty.")

    candidate_names = [kg_id]
    if not kg_id.endswith(".json"):
        candidate_names.append(f"{kg_id}.json")

    candidates: List[Path] = []
    for store_dir in _kg_store_candidates(request):
        for name in candidate_names:
            candidates.append(store_dir / name)

    path = _first_existing_path(candidates)
    if not path.exists():
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"KG id {kg_id!r} was not found. Tried: {tried}")
    return path


def resolve_knowledge_graph(request: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve KG input from inline KG, kg_path, or kg_id.

    Returns:
        A pair ``(kg_payload, kg_source_info)``.
    """
    req = _as_dict(request, "request")

    inline_kg = req.get("knowledge_graph") or req.get("kg") or req.get("graph_context")
    if inline_kg is not None:
        return inline_kg, {"kg_source": "inline"}

    kg_path = req.get("kg_path")
    if kg_path:
        resolved_candidates = _candidate_relative_paths(str(kg_path), req)
        resolved_path = _first_existing_path(resolved_candidates)
        kg = load_knowledge_graph_from_file(str(kg_path), request=req)
        return kg, {"kg_source": "kg_path", "kg_path": str(resolved_path)}

    kg_id = req.get("kg_id")
    if kg_id:
        resolved_path = resolve_kg_path_from_id(str(kg_id), req)
        kg = load_knowledge_graph_from_file(str(resolved_path), request=req)
        return kg, {"kg_source": "kg_id", "kg_id": str(kg_id), "kg_path": str(resolved_path)}

    raise ValueError(
        "Missing KG input. Provide one of: `knowledge_graph`, `kg`, `graph_context`, "
        "`kg_path`, or `kg_id`."
    )


def normalize_graph_context(knowledge_graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize several KG JSON shapes into SmartGenerator.graph_context.

    Supported inputs:
    1. ``{"nodes": [{"node_id": "n1", "content": "..."}],
        "relations": [{"subject": "A", "predicate": "HAS_STEP", "object": "B"}]}``
    2. ``{"triplets": [{"head": "A", "relation": "HAS_STEP", "tail": "B"}]}``
    3. ``{"edges": [...]}`` with either subject/predicate/object or head/relation/tail.
    """
    kg = _as_dict(knowledge_graph, "knowledge_graph")

    # Some callers pass {"knowledge_graph": {...}} directly.
    if "knowledge_graph" in kg and isinstance(kg["knowledge_graph"], dict):
        kg = kg["knowledge_graph"]

    nodes: List[Dict[str, Any]] = []
    for i, node in enumerate(kg.get("nodes", [])):
        if isinstance(node, str):
            content = node.strip()
            node_id = f"input_node_{i}"
        elif isinstance(node, dict):
            content = str(node.get("content") or node.get("text") or node.get("label") or "").strip()
            node_id = str(node.get("node_id") or node.get("id") or f"input_node_{i}")
        else:
            continue
        if content:
            nodes.append({"node_id": node_id, "content": content})

    raw_relations = kg.get("relations") or kg.get("edges") or kg.get("triplets") or []
    relations: List[Dict[str, Any]] = []
    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue
        subject = rel.get("subject", rel.get("head", rel.get("from", "")))
        predicate = rel.get("predicate", rel.get("relation", rel.get("type", "related")))
        obj = rel.get("object", rel.get("tail", rel.get("to", "")))
        if subject and obj:
            normalized = {
                "subject": str(subject),
                "predicate": str(predicate or "related"),
                "object": str(obj),
            }
            if rel.get("source_node"):
                normalized["source_node"] = rel["source_node"]
            relations.append(normalized)

    if not nodes and not relations:
        raise ValueError(
            "knowledge_graph must contain at least one non-empty node or relation/triplet."
        )

    return {"nodes": nodes, "relations": relations}


def extract_answer(question: Dict[str, Any], question_format: Optional[str] = None) -> Any:
    """Return a stable answer field regardless of question format."""
    q_format = question_format or question.get("question_format") or "mcq_single"
    if q_format == "mcq_single":
        return question.get("correct_answer")
    if q_format == "mcq_multi":
        return question.get("correct_answers")
    if q_format == "true_false":
        return question.get("tf_answer")
    if q_format == "fill_blank":
        return question.get("answers")
    if q_format == "open_answer":
        return question.get("model_answer")
    return question.get("answer") or question.get("correct_answer")


def generate_question_from_json(
    request: Dict[str, Any],
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
) -> Dict[str, Any]:
    """Generate one question from one JSON request.

    Minimal inline KG request:

    ``{"topic": "merge sort", "question_format": "mcq_single", "knowledge_graph": {...}}``

    Non-redundant file-based request:

    ``{"topic": "merge sort", "question_format": "mcq_single", "kg_path": "kg_store/merge_sort.json"}``

    Registry-id request:

    ``{"topic": "merge sort", "question_format": "mcq_single", "kg_id": "merge_sort"}``
    """
    req = _as_dict(request, "request")
    topic = str(req.get("topic") or "").strip()
    if not topic:
        raise ValueError("request.topic is required.")

    question_format = str(req.get("question_format") or "mcq_single").strip()
    if question_format not in VALID_QUESTION_FORMATS:
        raise ValueError(
            f"question_format must be one of {sorted(VALID_QUESTION_FORMATS)}, got {question_format!r}."
        )

    question_type = req.get("question_type")
    if question_type is not None:
        question_type = str(question_type).strip()
        if question_type not in VALID_QUESTION_TYPES:
            raise ValueError(
                f"question_type must be one of {sorted(VALID_QUESTION_TYPES)}, got {question_type!r}."
            )

    kg_payload, kg_source_info = resolve_knowledge_graph(req)
    graph_context = normalize_graph_context(kg_payload)

    resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not resolved_api_key:
        raise ValueError("DEEPSEEK_API_KEY is required, either in environment or as api_key.")

    # Import here so JSON normalization can be used without loading the full LLM stack.
    from .generator import SmartGenerator

    generator = SmartGenerator(api_key=resolved_api_key, base_url=base_url)
    start = time.time()
    raw_json, method = generator.generate(
        topic=topic,
        graph_context=graph_context,
        qb_retriever=None,
        question_format=question_format,
        question_type=question_type,
        naive_mode=bool(req.get("naive_mode", False)),
        no_self_correction=bool(req.get("no_self_correction", False)),
    )

    try:
        question = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Generator returned invalid JSON: {exc}. Raw output: {raw_json[:500]}") from exc

    question.setdefault("question_format", question_format)
    if question_type:
        question.setdefault("question_type", question_type)

    metadata: Dict[str, Any] = {
        "method": method,
        "model": "graph_rag_single_io",
        "generation_time": round(time.time() - start, 2),
        "kg_nodes": len(graph_context.get("nodes", [])),
        "kg_relations": len(graph_context.get("relations", [])),
    }
    metadata.update(kg_source_info)

    return {
        "topic": topic,
        "question_format": question_format,
        "question_type": question.get("question_type", question_type or "auto"),
        "question": question,
        "answer": extract_answer(question, question_format),
        "metadata": metadata,
    }
