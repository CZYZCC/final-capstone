import json
import statistics
from typing import List, Dict

from openai import OpenAI

from .logger          import Logger
from .knowledge_graph import AdvancedKnowledgeGraph
from .retriever       import VectorBaselineRetriever, LogicGraphRetriever, QuestionBankRetriever
from .generator       import NoRetrievalGenerator, BaselineGenerator, SmartGenerator
from .evaluator       import AutomatedEvaluator


class Pipeline:

    def __init__(
        self,
        api_key: str,
        output_dir: str,
        question_bank_path: str = "./question_bank.json",
    ):
        self.logger         = Logger(output_dir)
        self.kg             = AdvancedKnowledgeGraph(self.logger)
        self.qb_path        = question_bank_path

        llm_client          = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.no_ret_gen     = NoRetrievalGenerator(api_key)   # Group A
        self.vector_rag_gen = BaselineGenerator(api_key)       # Group B
        self.graph_rag_gen  = SmartGenerator(api_key)          # Group C
        self.evaluator      = AutomatedEvaluator(llm_client)

    def run(self, textbook_dir: str, triplets_path: str, topics: List[str]):
        self.kg.build_base_structure(textbook_dir)
        self.kg.load_triplets(triplets_path)

        vec_retriever   = VectorBaselineRetriever(self.kg)
        graph_retriever = LogicGraphRetriever(self.kg, vec_retriever)
        qb_retriever    = QuestionBankRetriever(self.qb_path, vec_retriever.encoder)

        if not qb_retriever.available:
            self.logger.log(
                "[!] question_bank.json not found — GRAPH_RAG will always use 'generated' branch."
            )

        self.evaluator.reset_session()

        results = []
        for topic in topics:
            self.logger.log(f"\n{'='*60}\n>>> Topic: {topic}\n{'='*60}")

            # Group A: NO_RETRIEVAL
            nr_json, _       = self.no_ret_gen.generate(topic, qb_retriever)  # <--- 修改这里：传入 qb_retriever
            nr_score         = self.evaluator.evaluate(nr_json, [], topic)
            self._log_and_save("NO_RETRIEVAL", topic, nr_json, nr_score, [], "no_retrieval")

            # Group B: VECTOR_RAG
            v_ctx            = vec_retriever.retrieve(topic)
            v_json, _        = self.vector_rag_gen.generate(topic, v_ctx, qb_retriever)  # <--- 修改这里：传入 qb_retriever
            v_score          = self.evaluator.evaluate(v_json, v_ctx, topic)
            self._log_and_save("VECTOR_RAG", topic, v_json, v_score, v_ctx, "vector_rag")

            # Group C: GRAPH_RAG
            # When the retrieved subgraph is sparse (< 20 relations), the graph alone
            # provides insufficient context. Rather than falling back entirely to
            # VECTOR_RAG, we enrich g_ctx by merging the vector-retrieved nodes in.
            # This preserves whatever graph relations exist while filling the content
            # gap with semantically relevant knowledge slices — giving the SmartGenerator
            # both the relational structure and enough factual grounding to work with.
            SPARSE_GRAPH_THRESHOLD = 20
            # MERGE_TOP_K must be larger than the graph retriever's internal seed
            # count (top_k=5) so we fetch nodes BEYOND the seeds already included
            # in g_ctx — otherwise dedup removes everything and merge adds 0 nodes.
            MERGE_TOP_K = 15
            g_ctx       = graph_retriever.retrieve_subgraph(topic)
            n_relations = len(g_ctx.get('relations', []))
            if n_relations < SPARSE_GRAPH_THRESHOLD:
                existing_ids   = {n['node_id'] for n in g_ctx.get('nodes', [])}
                extended_v_ctx = vec_retriever.retrieve(topic, top_k=MERGE_TOP_K)
                extra_nodes    = [
                    {'node_id': v['node_id'], 'content': v['content']}
                    for v in extended_v_ctx
                    if v['node_id'] not in existing_ids
                ]
                g_ctx = {
                    'nodes':     g_ctx.get('nodes', []) + extra_nodes,
                    'relations': g_ctx.get('relations', []),
                }
                self.logger.log(
                    f"   [GRAPH_RAG] Sparse subgraph ({n_relations} relations < "
                    f"{SPARSE_GRAPH_THRESHOLD}). Merged {len(extra_nodes)} vector nodes "
                    f"(top_k={MERGE_TOP_K}, excluding {len(existing_ids)} existing seeds) "
                    f"into graph context (total nodes: {len(g_ctx['nodes'])})."
                )
            g_json, g_method = self.graph_rag_gen.generate(topic, g_ctx, qb_retriever)
            g_score = self.evaluator.evaluate(g_json, g_ctx, topic)
            self._log_and_save("GRAPH_RAG", topic, g_json, g_score, g_ctx, g_method)

            results.append({
                "topic":        topic,
                "no_retrieval": nr_score.get("overall", 0),
                "vector_rag":   v_score.get("overall", 0),
                "graph_rag":    g_score.get("overall", 0),
                "method":       g_method,
            })

        self._print_final(results)

    # ------------------------------------------------------------------
    # FIX-L: format-aware logging
    # ------------------------------------------------------------------

    def _log_and_save(self, strategy, topic, q_json, score, ctx, method):
        try:
            data      = json.loads(q_json)
            q_type    = data.get("question_type", "unknown")
            q_fmt     = data.get("question_format", "mcq_single")
            div_label = score.get("diversity_label", "unknown")
            div_icon  = "💻" if "computational" in div_label else "📖"

            self.logger.log(
                f"   [{strategy}|{method}|{q_type}|{q_fmt}] "
                f"Overall:{score.get('overall', 0):.2f}/5.0 | "
                f"Rel:{score.get('relevance','?')} "
                f"Correct:{score.get('correctness','?')} "
                f"EC:{score.get('edge_case_triggering','?')} "
                f"MultiHop:{score.get('multi_hop_dependency','?')} "
                f"Diag:{score.get('diagnostic_power','?')} "
                f"GRD:{score.get('graph_relational_depth','?')} "
                f"Div:{score.get('diversity', 0):.1f}({div_icon}{div_label}) "
                f"DiffScore:{data.get('difficulty_score', 'N/A')}/5"
            )
            diff_reason = data.get("difficulty_reasoning", "")
            if diff_reason:
                self.logger.log(f"      [DifficultyJudge]: {diff_reason}")

            # FIX-L: format-aware question display
            if q_fmt == "true_false":
                stmt = data.get("statement", data.get("question", ""))
                ans  = data.get("tf_answer", "?")
                self.logger.log(f"      Statement: {stmt}")
                self.logger.log(f"      Answer: {ans}")
                expl = data.get("explanation", "")
                if expl:
                    self.logger.log(f"      Explanation: {expl[:200]}")

            elif q_fmt == "fill_blank":
                sent    = data.get("sentence", data.get("question", ""))
                answers = data.get("answers", [])
                self.logger.log(f"      Sentence: {sent}")
                self.logger.log(f"      Answers: {answers}")
                sp = data.get("scratchpad", "")
                if sp:
                    self.logger.log(f"      [Trace]: {sp[:200]}")

            elif q_fmt == "open_answer":
                self.logger.log(f"      Q: {data.get('question', '')}")
                model_ans = data.get("model_answer", "")
                if model_ans:
                    self.logger.log(f"      Model answer: {model_ans[:200]}")
                kps = data.get("key_points", [])
                for i, kp in enumerate(kps[:3]):
                    self.logger.log(f"      Key point {i+1}: {kp}")

            elif q_fmt == "mcq_multi":
                self.logger.log(f"      Q: {data.get('question', '')}")
                opts    = data.get("options", {})
                correct = data.get("correct_answers", [])
                for k, v in opts.items():
                    marker = " ✓" if k in correct else ""
                    self.logger.log(f"         {k}. {v}{marker}")
                expl = data.get("explanation", "")
                if expl:
                    self.logger.log(f"      Explanation: {expl[:200]}")

            else:  # mcq_single (default)
                self.logger.log(f"      Q: {data.get('question', '')}")
                for i, d in enumerate(data.get("distractors", [])[:3]):
                    if isinstance(d, dict):
                        self.logger.log(
                            f"         {chr(66+i)}. {d.get('option','')}  <-- {d.get('explanation','')}"
                        )
                    else:
                        self.logger.log(f"         {chr(66+i)}. {d}")

                sp = data.get("generator_scratchpad", {})
                if sp:
                    if q_type == "computational":
                        self.logger.log(
                            f"      [Trace]: {sp.get('step_by_step_execution', 'N/A')[:300]}"
                        )
                    else:
                        self.logger.log(f"      [Concept]: {sp.get('core_concept', 'N/A')}")
                self.logger.log(f"      Correct: {data.get('correct_answer', '')}")

            self.logger.save_artifact(
                f"{topic.replace(' ', '_')}_{strategy}.json",
                {"question": data, "score": score, "context": ctx, "method": method},
            )
        except Exception as e:
            self.logger.log(f"   [{strategy}] parse error: {e}")

    def _print_final(self, results: List[Dict]):
        self.logger.log(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
        self.logger.log(
            f"{'topic':<28} {'no_retrieval':>13} {'vector_rag':>11} {'graph_rag':>10}  method"
        )
        self.logger.log("-" * 70)

        for r in results:
            scores     = [r["no_retrieval"], r["vector_rag"], r["graph_rag"]]
            win_marker = " ✓" if r["graph_rag"] == max(scores) else ""
            self.logger.log(
                f"{r['topic']:<28} "
                f"{r['no_retrieval']:>13.2f} "
                f"{r['vector_rag']:>11.2f} "
                f"{r['graph_rag']:>10.2f}{win_marker}  "
                f"{r['method']}"
            )

        nr_avg = statistics.mean([r["no_retrieval"] for r in results])
        vr_avg = statistics.mean([r["vector_rag"]   for r in results])
        gr_avg = statistics.mean([r["graph_rag"]    for r in results])

        method_counts: Dict[str, int] = {}
        for r in results:
            method_counts[r["method"]] = method_counts.get(r["method"], 0) + 1

        self.logger.log("-" * 70)
        self.logger.log(
            f"{'AVERAGE':<28} {nr_avg:>13.2f} {vr_avg:>11.2f} {gr_avg:>10.2f}"
        )
        self.logger.log(f"\nGRAPH_RAG branch breakdown: {method_counts}")

        delta_vr           = vr_avg - nr_avg
        delta_gr           = gr_avg - nr_avg
        delta_graph_vs_vec = gr_avg - vr_avg

        self.logger.log(
            f"\nVector RAG  vs No-Retrieval : {delta_vr:+.2f} "
            f"({delta_vr/nr_avg*100 if nr_avg else 0:+.1f}%)"
        )
        self.logger.log(
            f"Graph RAG   vs No-Retrieval : {delta_gr:+.2f} "
            f"({delta_gr/nr_avg*100 if nr_avg else 0:+.1f}%)"
        )
        self.logger.log(
            f"Graph RAG   vs Vector RAG   : {delta_graph_vs_vec:+.2f} "
            f"({delta_graph_vs_vec/vr_avg*100 if vr_avg else 0:+.1f}%)"
        )

        if gr_avg > vr_avg > nr_avg:
            verdict = "✓ Full hierarchy confirmed: no_retrieval < vector_rag < graph_rag"
        elif gr_avg > nr_avg:
            verdict = "~ Graph RAG beats no-retrieval but not vector RAG"
        else:
            verdict = "✗ Graph RAG does not outperform no-retrieval baseline"

        self.logger.log(f"\n[RESULT] {verdict}")
        self.logger.log("=" * 70)
