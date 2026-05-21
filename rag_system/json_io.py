"""JSON input/output helpers for single-question generation.

This module converts a user-provided KG JSON into the graph_context format
expected by SmartGenerator, and wraps the raw generator output into one stable
JSON response object.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


VALID_QUESTION_FORMATS = {
    "mcq_single",
    "mcq_multi",
    "true_false",
    "fill_blank",
    "open_answer",
}

VALID_QUESTION_TYPES = {"computational", "conceptual"}


def _as_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object/dict.")
    return value


def normalize_graph_context(knowledge_graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize several KG JSON shapes into SmartGenerator.graph_context.

    Supported inputs:
    1. {"nodes": [{"node_id": "n1", "content": "..."}],
        "relations": [{"subject": "A", "predicate": "HAS_STEP", "object": "B"}]}
    2. {"triplets": [{"head": "A", "relation": "HAS_STEP", "tail": "B"}]}
    3. {"edges": [...]} with either subject/predicate/object or head/relation/tail.
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

    Request schema:
    {
      "topic": "merge sort",
      "question_format": "mcq_single",
      "question_type": "computational",       # optional: computational/conceptual
      "knowledge_graph": {
        "nodes": [{"node_id": "n1", "content": "..."}],
        "relations": [{"subject": "A", "predicate": "HAS_STEP", "object": "B"}]
      },
      "no_self_correction": false,              # optional
      "naive_mode": false                       # optional
    }
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

    kg_payload = req.get("knowledge_graph") or req.get("kg") or req.get("graph_context")
    if kg_payload is None:
        raise ValueError("request.knowledge_graph is required.")
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

    return {
        "topic": topic,
        "question_format": question_format,
        "question_type": question.get("question_type", question_type or "auto"),
        "question": question,
        "answer": extract_answer(question, question_format),
        "metadata": {
            "method": method,
            "model": "graph_rag_single_io",
            "generation_time": round(time.time() - start, 2),
            "kg_nodes": len(graph_context.get("nodes", [])),
            "kg_relations": len(graph_context.get("relations", [])),
        },
    }
