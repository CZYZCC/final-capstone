import json
from collections import Counter
from typing import List, Dict, Tuple, Set

from openai import OpenAI


# ---------------------------------------------------------------------------
# Topic classification sets (keep in sync with generator.py)
# ---------------------------------------------------------------------------

COMPUTATIONAL_TOPICS: Set[str] = {
    "hash table", "hash table linear probing", "hash table quadratic probing",
    "sorting", "merge sort", "quick sort", "insertion sort", "bubble sort",
    "heap sort", "heap", "binary heap",
    "graph traversal", "bfs", "dfs", "bfs graph traversal", "dfs graph traversal",
    "dynamic programming", "dynamic programming knapsack", "dynamic programming lcs",
    "recursion", "recursion and recurrence",
    "binary search", "binary search tree", "binary search tree operations",
    "dijkstra", "dijkstra shortest path",
    "data structures and algorithms", "data structure and algorithm",
    "mathematics", "mathematics (for computer science and ai)",
    "algorithm", "algorithms",
}

CONCEPTUAL_TOPICS: Set[str] = {
    "cybersecurity", "computer networks", "computer network",
    "information retrieval", "artificial intelligence introduction",
    "machine learning", "data science", "data science and big data",
    "human-computer interaction", "nlp", "natural language processing",
    "computer vision", "ethics", "programming fundamentals",
    "computer architecture", "operating systems",
    "database", "database systems",
}

# All valid question_format values (FIX-F)
ALL_FORMATS: Set[str] = {
    "mcq_single", "mcq_multi", "true_false", "fill_blank", "open_answer"
}


def _topic_expects_computational(topic: str) -> bool:
    t = topic.lower().strip()
    if any(ct in t for ct in COMPUTATIONAL_TOPICS):
        return True
    if any(ct in t for ct in CONCEPTUAL_TOPICS):
        return False
    return True


class AutomatedEvaluator:

    COMP_KEYWORDS = [
        'calculate', 'compute', 'trace', 'run through', 'what is the value',
        'how many steps', 'insert the following', 'delete from', 'find the index',
        'what will the', 'after applying', 'result of running', 'fill in the',
        'given the array', 'hash function', 'sort the following', 'apply',
        'what is the output', 'what does', 'state of', 'iteration', 'index',
        'collision', 'probing', 'complexity of', 'recurrence', 'base case',
        'given the sequence', 'after inserting', 'after deleting', 'show the',
        'how many comparisons', 'step by step', 'trace through', 'what is returned',
    ]

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client
        self._session_type_counts: Counter = Counter()

    def reset_session(self):
        self._session_type_counts.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, question_json: str, context, topic: str) -> Dict:
        try:
            q_data = json.loads(question_json)

            llm_scores = self._call_llm_judge(q_data, topic)

            r   = float(llm_scores.get('relevance',            3))
            cr  = float(llm_scores.get('correctness',          3))
            dp  = float(llm_scores.get('diagnostic_power',     3))
            mh  = float(llm_scores.get('multi_hop_dependency', 3))
            ec  = float(llm_scores.get('edge_case_triggering', 3))
            grd = float(llm_scores.get('graph_relational_depth', 3))

            div_label, div_score = self._assess_diversity(
                question_text   = q_data.get('question', q_data.get('statement', q_data.get('sentence', ''))),
                question_type   = q_data.get('question_type', ''),
                question_format = q_data.get('question_format', 'mcq_single'),
                topic           = topic,
            )

            overall = (r   * 0.05
                     + div_score * 0.10
                     + cr  * 0.20
                     + dp  * 0.20
                     + mh  * 0.15
                     + ec  * 0.20
                     + grd * 0.10)

            return {
                'relevance':              r,
                'correctness':            cr,
                'diagnostic_power':       dp,
                'multi_hop_dependency':   mh,
                'edge_case_triggering':   ec,
                'graph_relational_depth': grd,
                'diversity':              div_score,
                'diversity_label':        div_label,
                'overall':                overall,
                'details':                llm_scores,
            }

        except Exception as e:
            return {'overall': 0, 'diversity_label': 'unknown', 'error': str(e)}

    def evaluate_batch(
        self,
        items: List[Dict],
        context_map: Dict[str, List[Dict]],
        topic_map: Dict[str, str],
    ) -> List[Dict]:
        scored = []
        for item in items:
            qid   = item['id']
            score = self.evaluate(
                item['question_json'],
                context_map.get(qid, []),
                topic_map.get(qid, ''),
            )
            score['id'] = qid
            scored.append(score)
        return scored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm_judge(self, q_data: Dict, topic: str) -> Dict:
        """
        FIX-E: format-aware LLM judge.
        Detects question_format and adapts the rubric accordingly.
        """
        # Determine question format
        fmt = q_data.get('question_format', 'mcq_single')
        # Legacy questions without question_format field default to mcq_single
        if fmt not in ALL_FORMATS:
            fmt = 'mcq_single'

        # Extract fields depending on format
        actual_q_data = q_data.get('mcq_data', q_data)
        scratchpad = actual_q_data.get(
            'generator_scratchpad',
            q_data.get('generator_scratchpad', {}),
        )

        # Build format-specific question block
        if fmt == 'true_false':
            question_block = f"""Statement:     {actual_q_data.get('statement', q_data.get('statement', ''))}
Correct answer: {'True' if actual_q_data.get('tf_answer', q_data.get('tf_answer')) else 'False'}
Explanation:   {actual_q_data.get('explanation', '')}"""
        elif fmt == 'fill_blank':
            question_block = f"""Sentence:      {actual_q_data.get('sentence', q_data.get('sentence', ''))}
Answers:       {json.dumps(actual_q_data.get('answers', q_data.get('answers', [])))}
Explanation:   {actual_q_data.get('explanation', '')}"""
        elif fmt == 'open_answer':
            question_block = f"""Question:      {actual_q_data.get('question', '')}
Model answer:  {actual_q_data.get('model_answer', q_data.get('model_answer', ''))}
Key points:    {json.dumps(actual_q_data.get('key_points', q_data.get('key_points', [])))}"""
        elif fmt == 'mcq_multi':
            question_block = f"""Question:         {actual_q_data.get('question', '')}
Options:          {json.dumps(actual_q_data.get('options', {}))}
Correct answers:  {json.dumps(actual_q_data.get('correct_answers', []))}
Explanation:      {actual_q_data.get('explanation', '')}"""
        else:  # mcq_single (default)
            question_block = f"""Question:         {actual_q_data.get('question', '')}
Correct answer:   {actual_q_data.get('correct_answer', '')}
Question type:    {actual_q_data.get('question_type', 'unknown')}
Generator scratchpad: {json.dumps(scratchpad)}
Distractors:      {json.dumps(actual_q_data.get('distractors', []))}"""

        # Format-specific diagnostic_power rubric (Strict 1.0-5.0 Mapping)
        if fmt in ['mcq_single', 'mcq_multi']:
            dp_rubric = """diagnostic_power (1.0-5.0):
  1.0 - Random Noise: Distractors are generated randomly with no logic.
  2.0 - Surface Plausibility: Options look plausible (e.g., similar numbers) but lack specific error traceability.
  3.0 - Arithmetic/Typo Error: Corresponds to simple arithmetic mistakes or basic off-by-one calculation errors.
  4.0 - Procedural Omission: Perfectly corresponds to forgetting a specific algorithmic step (e.g., forgetting DP base case initialization, or ignoring isolated nodes in graph traversal).
  5.0 - Cognitive Misconception: Precisely matches the result of confusing Algorithm A with Algorithm B (e.g., using a stack instead of a queue for BFS)."""
        elif fmt == 'true_false':
            dp_rubric = """diagnostic_power (1.0-5.0):
  1.0-2.0: statement is a vague generalisation or obviously true/false.
  4.0-5.0: statement makes a specific, verifiable claim about a mechanism, consequence, or trade-off that requires deep understanding to evaluate."""
        elif fmt == 'fill_blank':
            dp_rubric = """diagnostic_power (1.0-5.0):
  1.0-2.0: blanks can be filled by guessing or by looking up a single keyword.
  4.0-5.0: each blank requires computing or reasoning through a specific step; wrong answers would arise from specific traceable errors."""
        else:  # open_answer
            dp_rubric = """diagnostic_power (1.0-5.0):
  1.0-2.0: key points are vague or restate the question.
  4.0-5.0: key points specify precise technical criteria a student must address."""

        prompt = f"""You are an elite Computer Science pedagogy expert evaluating a generated exam question.
Grade this question (format: {fmt}) STRICTLY according to the following rubrics.

=== SCORING DIMENSIONS (1.0 - 5.0) ===

relevance (1.0-5.0):
  Is the question strictly about "{topic}"? (1.0 = completely off-topic, 5.0 = perfectly targeted)

correctness (1.0-5.0):
  Is the correct answer truly correct and the math/logic flawless? (1.0 = totally wrong, 5.0 = mathematically perfect)

{dp_rubric}

multi_hop_dependency (1.0-5.0):
  1.0 - Single-Step: Just applying a formula or definition, no intermediate state tracking.
  2.0 - Shallow Trace: Algorithm executes only 1-2 steps.
  3.0 - Standard Algorithmic Trace: Completely traces a single independent algorithm where states update in a loop.
  4.0 - Mechanism Interaction: Triggers the interaction of two different mechanisms within the same algorithm.
  5.0 - Cross-Concept Synthesis: Output of an upstream concept MUST be the input of a downstream concept (e.g., DFS output used as DP input).

edge_case_triggering (1.0-5.0):
  1.0 - Happy Path: Most basic input, triggers no special branches.
  2.0 - Average Case: Random input with no special pattern (e.g., standard unsorted array).
  3.0 - Standard Edge Case: Triggers a specific conditional branch (e.g., duplicates to test stability, or hash collisions).
  4.0 - Degeneration Test: Input forces the algorithm into its theoretical worst-case complexity (e.g., reverse-sorted array for quicksort).
  5.0 - Compound Trap: Combines multiple anomalous states simultaneously (e.g., graph with isolated nodes, self-loops, and multi-edges).

graph_relational_depth (1.0-5.0):
  1.0 - No Grounding: Relies purely on LLM parametric memory.
  2.0 - Keyword Borrowing: Mentions terms from context, but logic doesn't depend on them.
  3.0 - Fact Extraction: Relies on a single specific fact/formula from context.
  4.0 - Relation Usage: Utilizes relationship edges from the graph to set constraints.
  5.0 - Topology Reasoning: Perfectly synthesizes a multi-hop logical chain strictly derived from the provided graph context.

=== QUESTION TO EVALUATE ===
Format: {fmt}
{question_block}

Return ONLY valid JSON (no markdown):
{{
  "verification_audit": "Solve the problem independently step-by-step. Analyze distractors.",
  "verdict": "CORRECT or MINOR_ERROR or WRONG",
  "critique_feedback": "Briefly critique the question against the rigorous rubrics above.",
  "correctness": <float 1.0-5.0>,
  "relevance": <float 1.0-5.0>,
  "diagnostic_power": <float 1.0-5.0>,
  "multi_hop_dependency": <float 1.0-5.0>,
  "edge_case_triggering": <float 1.0-5.0>,
  "graph_relational_depth": <float 1.0-5.0>
}}"""

        res = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500          # raised from 800 — long verification_audit can exceed 800
        )
        raw = res.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Response was truncated mid-JSON (exceeded max_tokens).
            # Return safe fallback scores so the pipeline never records 0.
            import re as _re
            scores: Dict = {}
            for key in ("correctness", "relevance", "diagnostic_power",
                        "multi_hop_dependency", "edge_case_triggering",
                        "graph_relational_depth"):
                m = _re.search(rf'"{key}"\s*:\s*([0-9.]+)', raw)
                scores[key] = float(m.group(1)) if m else 3.0
            scores["verdict"]            = "TRUNCATED"
            scores["verification_audit"] = raw[:300] + "…[truncated]"
            scores["critique_feedback"]  = "Response was truncated; partial scores extracted."
            return scores

    def _assess_diversity(
        self,
        question_text: str,
        question_type: str,
        question_format: str,
        topic: str,
    ) -> Tuple[str, float]:
        """
        FIX-3 + FIX-4 + FIX-F:

        1. TYPE APPROPRIATENESS (0–5)
           Does the question type fit the topic?
           Computational topic + computational question → 5.0
           Conceptual topic   + conceptual question   → 5.0
           Mismatch → 1.0 (computational) or 3.0 (conceptual-for-algo)

        2. FORMAT NOVELTY (FIX-F)
           Each (topic, question_format) pair is tracked separately.
           A True/False question and an MCQ on the same topic count as
           DIFFERENT format types, giving full diversity credit.

        3. SESSION NOVELTY PENALTY
           Same (topic, format) pair seen ≥2 times → small penalty.
        """
        topic_norm   = topic.lower().strip()
        expects_comp = _topic_expects_computational(topic)

        # Determine actual question type
        actual_is_comp = False
        if question_type == 'computational':
            actual_is_comp = True
        else:
            q_lower = question_text.lower()
            hits    = sum(1 for kw in self.COMP_KEYWORDS if kw in q_lower)
            actual_is_comp = (hits >= 3)

        actual_type_label = "computational" if actual_is_comp else "conceptual"

        # Type appropriateness score
        if expects_comp and actual_is_comp:
            base_score = 5.0
            label      = "computational_appropriate"
        elif (not expects_comp) and (not actual_is_comp):
            base_score = 5.0
            label      = "conceptual_appropriate"
        elif expects_comp and (not actual_is_comp):
            base_score = 3.0
            label      = "conceptual_for_algo_topic"
        else:
            base_score = 1.0
            label      = "computational_mismatch"

        # FIX-F: track (topic, question_format) not just (topic, type)
        # so different formats on the same topic each get full novelty credit
        fmt_label    = question_format if question_format in ALL_FORMATS else "mcq_single"
        session_key  = (topic_norm, fmt_label)
        self._session_type_counts[session_key] += 1
        repeat_count = self._session_type_counts[session_key]

        novelty_penalty = 0.0
        if repeat_count == 2:
            novelty_penalty = 0.5
        elif repeat_count >= 3:
            novelty_penalty = 1.0

        final_score = max(1.0, base_score - novelty_penalty)
        # Append format info to label for logging
        full_label = f"{label}|{fmt_label}"
        return full_label, final_score