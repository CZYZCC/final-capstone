"""
run_format_test.py
Small-scale test: generate questions in specific formats for specific topics.
Run this BEFORE the large-scale experiment to verify format support works.

Usage:
    python run_format_test.py

Edit TEST_CASES below to choose any combination of:
    topic / question_format / question_type / model
"""

import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

from rag_system import Pipeline
from rag_system.knowledge_graph import AdvancedKnowledgeGraph
from rag_system.retriever import (VectorBaselineRetriever, LogicGraphRetriever,
                                   QuestionBankRetriever)
from rag_system.generator import (NoRetrievalGenerator, BaselineGenerator,
                                   SmartGenerator)
from rag_system.evaluator import AutomatedEvaluator
from rag_system.logger import Logger
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================
API_KEY            = os.getenv("DEEPSEEK_API_KEY")
TEXTBOOK_DIR       = "./GraphRAG-Bench/textbooks"
TRIPLETS_PATH      = "./global_knowledge_graph.json"
QUESTION_BANK_PATH = "./question_bank.json"

# ── Define your test cases here ──────────────────────────────
# Each entry: (topic, question_format, question_type, model)
# model: "no_retrieval" | "vector_rag" | "graph_rag"
# question_format: "mcq_single"|"mcq_multi"|"true_false"|"fill_blank"|"open_answer"
# question_type:  "computational" | "conceptual"
TEST_CASES = [
    ("recursion",       "fill_blank",   "computational", "graph_rag"),
    ("recursion",       "fill_blank",   "computational", "vector_rag"),
    ("recursion",       "fill_blank",   "computational", "no_retrieval"),
]
# ──────────────────────────────────────────────────────────────

SPARSE_GRAPH_THRESHOLD = 20
MERGE_TOP_K            = 15


def setup():
    """Load KG and all retrievers/generators once."""
    log_dir = "./experiment_logs/format_test_" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path  = os.path.join(log_dir, "format_test_results.jsonl")
    print(f"Loading knowledge graph and retrievers…")
    print(f"Results will be saved to: {log_path}")
    logger = Logger(log_dir)
    kg     = AdvancedKnowledgeGraph(logger)
    kg.build_base_structure(TEXTBOOK_DIR)
    kg.load_triplets(TRIPLETS_PATH)

    vec_retriever   = VectorBaselineRetriever(kg)
    graph_retriever = LogicGraphRetriever(kg, vec_retriever)
    qb_retriever    = QuestionBankRetriever(QUESTION_BANK_PATH, vec_retriever.encoder)

    llm_client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    no_ret_gen     = NoRetrievalGenerator(API_KEY)
    vector_rag_gen = BaselineGenerator(API_KEY)
    graph_rag_gen  = SmartGenerator(API_KEY)
    evaluator      = AutomatedEvaluator(llm_client)

    print("Ready.\n")
    return (vec_retriever, graph_retriever, qb_retriever,
            no_ret_gen, vector_rag_gen, graph_rag_gen, evaluator, log_path)


def _build_graph_ctx(graph_retriever, vec_retriever, topic):
    """Retrieve graph context with sparse-graph merge."""
    g_ctx       = graph_retriever.retrieve_subgraph(topic)
    n_relations = len(g_ctx.get("relations", []))
    if n_relations < SPARSE_GRAPH_THRESHOLD:
        existing_ids   = {n["node_id"] for n in g_ctx.get("nodes", [])}
        extended       = vec_retriever.retrieve(topic, top_k=MERGE_TOP_K)
        extra          = [{"node_id": v["node_id"], "content": v["content"]}
                          for v in extended if v["node_id"] not in existing_ids]
        g_ctx = {"nodes": g_ctx.get("nodes", []) + extra,
                 "relations": g_ctx.get("relations", [])}
        print(f"    [sparse graph: {n_relations} rel, merged {len(extra)} vector nodes]")
    return g_ctx


def run_test(idx, topic, question_format, question_type, model,
             vec_retriever, graph_retriever, qb_retriever,
             no_ret_gen, vector_rag_gen, graph_rag_gen, evaluator,
             log_path: str = None):

    print(f"{'='*60}")
    print(f"Test {idx}: topic='{topic}' | format={question_format} "
          f"| type={question_type} | model={model}")
    print(f"{'='*60}")

    # Generate
    t0 = time.time()
    if model == "no_retrieval":
        q_json, method = no_ret_gen.generate(
            topic, qb_retriever, question_format=question_format)

    elif model == "vector_rag":
        v_ctx  = vec_retriever.retrieve(topic, top_k=8)
        q_json, method = vector_rag_gen.generate(
            topic, v_ctx, qb_retriever, question_format=question_format)

    else:  # graph_rag
        g_ctx  = _build_graph_ctx(graph_retriever, vec_retriever, topic)
        q_json, method = graph_rag_gen.generate(
            topic, g_ctx, qb_retriever, question_format=question_format)

    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s  (method={method})")

    # Parse and display
    try:
        data = json.loads(q_json)
    except Exception:
        print(f"  [ERROR] Could not parse JSON:\n  {q_json[:300]}")
        return

    _display_question(data, question_format)

    # Evaluate
    print("\n  Evaluating…")
    score = evaluator.evaluate(q_json, [], topic)
    _display_score(score)

    # Save to JSONL
    if log_path:
        record = {
            "test_idx":        idx,
            "topic":           topic,
            "question_format": question_format,
            "question_type":   question_type,
            "model":           model,
            "method":          method,
            "generation_time": round(gen_time, 2),
            "question":        data,
            "score":           score,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [Saved to {os.path.basename(log_path)}]")
    print()


def _display_question(data: dict, fmt: str):
    """Print the generated question in a readable way."""
    print()
    if fmt == "mcq_single":
        print(f"  Q: {data.get('question', '')}")
        for d in data.get("distractors", [])[:3]:
            opt = d.get("option", d) if isinstance(d, dict) else d
            print(f"    ✗ {opt}")
        print(f"  ✓ {data.get('correct_answer', '')}")

    elif fmt == "mcq_multi":
        print(f"  Q: {data.get('question', '')}")
        opts = data.get("options", {})
        correct = set(data.get("correct_answers", []))
        for k, v in opts.items():
            mark = "✓" if k in correct else "✗"
            print(f"    {mark} {k}. {v}")

    elif fmt == "true_false":
        ans = data.get("tf_answer", "?")
        print(f"  Statement: {data.get('statement', '')}")
        print(f"  Answer: {'TRUE' if ans else 'FALSE'}")
        print(f"  Explanation: {data.get('explanation', '')[:200]}")

    elif fmt == "fill_blank":
        print(f"  Sentence: {data.get('sentence', '')}")
        print(f"  Answers:  {data.get('answers', [])}")
        print(f"  Explanation: {data.get('explanation', '')[:150]}")

    elif fmt == "open_answer":
        print(f"  Q: {data.get('question', '')}")
        print(f"  Model answer: {data.get('model_answer', '')[:250]}…")
        kps = data.get("key_points", [])
        for i, kp in enumerate(kps[:4], 1):
            print(f"  Key point {i}: {kp}")


def _display_score(score: dict):
    overall = score.get("overall", 0)
    print(f"  Score: {overall:.2f}/5.0  |  "
          f"Correct:{score.get('correctness','?')}  "
          f"Rel:{score.get('relevance','?')}  "
          f"MH:{score.get('multi_hop_dependency','?')}  "
          f"EC:{score.get('edge_case_triggering','?')}  "
          f"GRD:{score.get('graph_relational_depth','?')}")
    if score.get("error"):
        print(f"  [Eval error] {score['error']}")


def main():
    (vec_retriever, graph_retriever, qb_retriever,
     no_ret_gen, vector_rag_gen, graph_rag_gen, evaluator, log_path) = setup()

    for i, (topic, fmt, qtype, model) in enumerate(TEST_CASES, 1):
        run_test(i, topic, fmt, qtype, model,
                 vec_retriever, graph_retriever, qb_retriever,
                 no_ret_gen, vector_rag_gen, graph_rag_gen, evaluator,
                 log_path=log_path)

    print("All format tests done.")


if __name__ == "__main__":
    main()
