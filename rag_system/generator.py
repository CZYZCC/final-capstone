import json
import random
import re
from typing import List, Dict, Tuple, Optional

from openai import OpenAI

try:
    from .rv_generator import generate_rv_question
    _RV_AVAILABLE = True
except ImportError:
    _RV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Difficulty Assessment (LLM Judge) — 1-5 numeric scale
# ---------------------------------------------------------------------------

MAX_DIFFICULTY_RETRIES = 3   # max regeneration attempts per question
DIFFICULTY_THRESHOLD   = 3   # scores <= this trigger regeneration (1 and 2 are rejected)

_DIFFICULTY_JUDGE_PROMPT = """\
You are an expert Computer Science exam psychometrician. Score the difficulty of the \
following MCQ for university-level CS students (sophomore / junior year) on a strict \
INTEGER scale from 1 to 5.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{correct_answer}

=== RATIONALE ===
{rationale}

=== STRICT SCORING RUBRIC ===

Score 1 — TRIVIAL
  • Pure definition recall with no computation.
  • The answer can be copied verbatim from a one-sentence textbook glossary entry.
  • Examples: "What data structure does BFS use?", "Define a base case.",
    "Which sorting algorithm has O(n log n) average time complexity?"

Score 2 — SIMPLE
  • Single-algorithm mechanical trace on a small, happy-path input (≤ 6 elements).
  • OR one-step formula substitution (the formula is directly named or universally known).
  • The student follows one rule with no branching, no edge condition, no state
    accumulation across more than 4 steps.
  • Examples: "How many comparisons does insertion sort make on a reverse-sorted
    N-element array?" (answer = N*(N-1)/2), "Trace BFS on a 3-node graph from A.",
    "Apply h(k) = k mod 7 to k = 14."

Score 3 — MODERATE
  • Multi-step trace (≥ 5 distinct state-changing operations) requiring sustained
    attention to detail, OR a single algorithm applied to input with ONE non-trivial
    edge condition (collision, boundary, missing base case, etc.).
  • The student must track evolving state but uses a single concept without surprises.
  • Most carefully-studying students can answer; careless ones make errors.

Score 4 — CHALLENGING  (EITHER of the two paths below qualifies)
  PATH A — Cross-concept: Requires integrating exactly TWO distinct algorithmic
    concepts that CAUSALLY INTERACT (e.g. hash function output → collision chain
    length, pivot choice → recursion depth, base-case design → space complexity).
    The student must reason across the boundary between the two concepts.
  PATH B — Deep single-concept adversarial: Uses ONE algorithm, but ALL of:
    (a) ≥ 10 distinct state-changing operations in a full mental simulation,
    (b) an adversarial or degenerate input that forces worst-case / boundary behavior,
    (c) at least ONE non-obvious trap — a specific step where even careful students
        commonly make a wrong assumption (must be identifiable in the question),
    (d) the correct answer CANNOT be reached by rule-of-thumb or formula alone;
        full step-by-step simulation is required.
  Well-prepared students find it hard; less-prepared ones succeed < 50% of the time.

Score 5 — EXPERT  (EITHER of the two paths below qualifies)
  PATH A — Multi-concept: Synthesises THREE or more concepts with non-obvious
    causal interactions or cross-domain reasoning.
  PATH B — Deep single-concept: ONE algorithm, but the question requires the student
    to simultaneously reason at TWO distinct levels — e.g. trace the algorithm's
    concrete execution AND diagnose why a correctness or complexity claim is violated,
    or identify BOTH the algorithm's output AND the invariant it breaks, in a single
    question. The correct answer contradicts the intuition of most prepared students.

=== SCORING DISCIPLINE ===
- Assign score 2 if the answer follows from a named formula or a ≤ 4-step trace,
  regardless of how the question is phrased.
- Assign score 3 only when ≥ 5 distinct state-changing operations must be executed.
- Assign score 4 via PATH A only when you can name BOTH concepts and state the causal
  direction. Assign score 4 via PATH B only when you can identify the specific
  non-obvious trap AND confirm the trace requires ≥ 10 operations.
- Default to the LOWER score when uncertain between two adjacent levels.

Return ONLY valid JSON, no markdown:
{{"score": <integer 1-5>, "reasoning": "2-3 sentence explanation citing the specific \
path (A or B) and the concrete evidence from the question text"}}
"""


def assess_difficulty(llm_client: OpenAI, question_json_str: str) -> Tuple[int, str]:
    """
    Call LLM to judge the difficulty of a generated question on a 1-5 scale.

    Returns
    -------
    (score, reasoning)
        score     : int 1-5  (defaults to 3 on error — treated as acceptable)
        reasoning : brief human-readable explanation from the judge LLM
    """
    try:
        q = json.loads(question_json_str)
    except Exception:
        return 3, "JSON parse error — treating as acceptable (score=3)"

    question       = q.get("question", "")
    correct_answer = q.get("correct_answer", "")
    rationale      = q.get("rationale", "")[:500]

    if not question:
        return 3, "Empty question — treating as acceptable (score=3)"

    prompt = _DIFFICULTY_JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        rationale=rationale,
    )

    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0,
        )
        result    = json.loads(response.choices[0].message.content)
        raw_score = result.get("score", 3)
        reasoning = result.get("reasoning", "")
        try:
            score = int(raw_score)
        except (ValueError, TypeError):
            score = 3
        score = max(1, min(5, score))   # clamp to [1, 5]
        return score, reasoning
    except Exception as exc:
        return 3, f"Judge LLM error ({exc}) — treating as acceptable (score=3)"


# ---------------------------------------------------------------------------
# Difficulty Boost Block (injected into retry prompts)
# ---------------------------------------------------------------------------

_DIFFICULTY_BOOST_TEMPLATE = """\
╔══════════════════════════════════════════════════════════╗
║            ⚠  DIFFICULTY BOOST REQUIRED  ⚠              ║
╚══════════════════════════════════════════════════════════╝

The question below was REJECTED by an automated difficulty reviewer.
It received a score of {score}/5, which is at or below the minimum
acceptable threshold of {threshold}/5:

--- REJECTED QUESTION ---
{rejected_question}
--- END REJECTED QUESTION ---

REVIEWER'S REASON FOR LOW SCORE:
  "{judge_reasoning}"

You MUST produce a question that scores AT LEAST {threshold}/5.
The following are STRICTLY FORBIDDEN:
  ✗  Score 1: pure definition recall
  ✗  Score 2: direct formula substitution, trace on ≤ 6 happy-path elements,
     or any question structurally similar to the rejected one above

To reach score 4, choose ONE of these two valid paths:

  PATH A — Cross-concept (easier if you have graph context):
    Combine TWO distinct algorithmic concepts that causally interact.
    Name both explicitly. Example: "how does the hash function's output
    determine the length of the linear-probing collision chain?"

  PATH B — Deep single-concept adversarial (works without graph context):
    Use ONE algorithm, but satisfy ALL four conditions:
    (a) The student must execute ≥ 10 distinct state-changing steps
    (b) Use an adversarial or degenerate input (worst-case, all-equal,
        reverse-sorted, degenerate tree, all-collision hash keys, etc.)
    (c) Embed ONE non-obvious trap — a specific step where even careful
        students commonly make a wrong assumption. State the trap clearly
        in generator_scratchpad.edge_case_justification.
    (d) The correct answer must be unreachable without full simulation;
        no formula or rule-of-thumb can shortcut to the answer.

"""


def _build_difficulty_boost(rejected_q_json: str, judge_score: int, judge_reasoning: str) -> str:
    """
    Build the difficulty-boost prefix that is prepended to the generation
    prompt when the previous attempt scored below DIFFICULTY_THRESHOLD.
    """
    try:
        q = json.loads(rejected_q_json)
        rejected_question = q.get("question", "")[:400]
    except Exception:
        rejected_question = str(rejected_q_json)[:400]
    return _DIFFICULTY_BOOST_TEMPLATE.format(
        score=judge_score,
        threshold=DIFFICULTY_THRESHOLD,
        rejected_question=rejected_question,
        judge_reasoning=judge_reasoning,
    )


# ---------------------------------------------------------------------------
# Topic classification (mirrors evaluator.py — keep in sync)
# ---------------------------------------------------------------------------

_COMPUTATIONAL_TOPICS = {
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

_CONCEPTUAL_TOPICS = {
    "cybersecurity", "computer networks", "computer network",
    "information retrieval", "artificial intelligence introduction",
    "machine learning", "data science", "data science and big data",
    "human-computer interaction", "nlp", "natural language processing",
    "computer vision", "ethics", "programming fundamentals",
    "computer architecture", "operating systems",
    "database", "database systems",
}


def _choose_question_type(topic: str) -> str:
    t = topic.lower().strip()
    if any(ct in t for ct in _COMPUTATIONAL_TOPICS):
        return "computational"
    if any(ct in t for ct in _CONCEPTUAL_TOPICS):
        return "conceptual"
    return "computational"


# ---------------------------------------------------------------------------
# One-shot example pools
# ---------------------------------------------------------------------------

_ONESHOT_POOL = [
    {
        "generator_scratchpad": {
            "algorithm_rules": "Hash function h(k) = k mod 10, Linear Probing.",
            "step_by_step_execution": (
                "h(12)=2, empty → insert at 2. "
                "h(22)=2, occupied → probe 3, empty → insert at 3. "
                "h(32)=2, occupied → probe 3, occupied → probe 4, empty → insert at 4."
            ),
            "final_state": "Index 4",
        },
        "mcq_data": {
            "question": (
                "A hash table uses linear probing with h(k) = k mod 10. "
                "Insert keys [12, 22, 32] in order. Which index does 32 occupy?"
            ),
            "correct_answer": "Index 4",
            "rationale": "h(12)=2. h(22)=2 collides → 3. h(32)=2 collides → 3 collides → 4.",
            "distractors": [
                {"option": "Index 2", "explanation": "[Initialization Error] Forgetting 12 already occupies slot 2."},
                {"option": "Index 3", "explanation": "[Procedural Omission] Stopping one probe too early."},
                {"option": "Index 5", "explanation": "[Operator Confusion] Applying double-hashing step instead of linear probing."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "Merge sort: recursively split, sort, merge.",
            "step_by_step_execution": (
                "Array [5,2,4,1]. Split → [5,2] and [4,1]. "
                "Sort [5,2] → merge [2,5]. Sort [4,1] → merge [1,4]. "
                "Merge [2,5] and [1,4]: compare 2 vs 1→1; 2 vs 4→2; 5 vs 4→4; 5. Result [1,2,4,5]."
            ),
            "final_state": "[1, 2, 4, 5]",
        },
        "mcq_data": {
            "question": (
                "Apply merge sort to [5, 2, 4, 1]. "
                "What is the final sorted array after all merge steps?"
            ),
            "correct_answer": "[1, 2, 4, 5]",
            "rationale": "Split into [5,2]→[2,5] and [4,1]→[1,4]. Merging gives [1,2,4,5].",
            "distractors": [
                {"option": "[1, 2, 5, 4]", "explanation": "[Procedural Omission] Merging stops one step early."},
                {"option": "[2, 4, 1, 5]", "explanation": "[Initialization Error] Sorted only the first half."},
                {"option": "[1, 4, 2, 5]", "explanation": "[Boundary Error] Incorrect merge comparison."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "BFS uses a queue; visit in FIFO order, mark visited before enqueuing.",
            "step_by_step_execution": (
                "Graph: A-B, A-C, B-D, C-D. Start at A. "
                "Enqueue A, visited={A}. Dequeue A → enqueue B,C. "
                "Dequeue B → enqueue D. Dequeue C → D already visited. Dequeue D. "
                "Visit order: A, B, C, D."
            ),
            "final_state": "A, B, C, D",
        },
        "mcq_data": {
            "question": (
                "Run BFS on the graph with edges A–B, A–C, B–D, C–D starting from node A "
                "(neighbours processed alphabetically). What is the visit order?"
            ),
            "correct_answer": "A, B, C, D",
            "rationale": "Queue: [A]→dequeue A, enqueue B,C → [B,C]→dequeue B, enqueue D → [C,D]→dequeue C (D visited)→dequeue D.",
            "distractors": [
                {"option": "A, B, D, C", "explanation": "[Procedural Omission] Processing C after D instead of FIFO."},
                {"option": "A, C, B, D", "explanation": "[Initialization Error] Reversing neighbour order."},
                {"option": "A, D, B, C", "explanation": "[Operator Confusion] Confusing BFS queue with a DFS stack."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "0/1 knapsack DP: dp[i][w] = max(dp[i-1][w], dp[i-1][w-wi]+vi) if wi<=w.",
            "step_by_step_execution": (
                "Items: (w=2,v=3), (w=3,v=4). Capacity W=4. "
                "dp[1][2]=3, dp[1][3]=3, dp[1][4]=3. "
                "dp[2][4]=max(dp[1][4], dp[1][1]+4)=max(3,4)=4."
            ),
            "final_state": "4",
        },
        "mcq_data": {
            "question": (
                "Apply the 0/1 knapsack algorithm. Items: item1(weight=2, value=3), item2(weight=3, value=4). "
                "Knapsack capacity W=4. What is the maximum value achievable?"
            ),
            "correct_answer": "4",
            "rationale": "Only item2 fits alone (value=4). Both together need weight 5>4. Maximum is 4.",
            "distractors": [
                {"option": "7", "explanation": "[Boundary Error] Adding both values without checking total weight."},
                {"option": "3", "explanation": "[Procedural Omission] Selecting only item1."},
                {"option": "6", "explanation": "[Operator Confusion] Taking item1 twice as unbounded knapsack."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
    {
        "generator_scratchpad": {
            "algorithm_rules": "BST insertion: go left if key < current node, right if key > current node.",
            "step_by_step_execution": (
                "Insert [10, 5, 15, 3] into empty BST. "
                "10 → root. 5<10 → left child of 10. 15>10 → right child of 10. "
                "3<10 → go left to 5; 3<5 → left child of 5."
            ),
            "final_state": "3 is the left child of 5",
        },
        "mcq_data": {
            "question": (
                "Insert keys [10, 5, 15, 3] one by one into an initially empty BST. "
                "Where is node 3 located in the final tree?"
            ),
            "correct_answer": "Left child of 5",
            "rationale": "10 is root. 5 goes left of 10. 3<10→go left to 5; 3<5→left child of 5.",
            "distractors": [
                {"option": "Right child of 5", "explanation": "[Boundary Error] Choosing right ignoring key comparison 3<5."},
                {"option": "Left child of 10", "explanation": "[Procedural Omission] Stopping at 10's left without continuing."},
                {"option": "Left child of 15", "explanation": "[Operator Confusion] Routing based on insertion order."},
            ],
            "question_type": "computational",
            "source": "oneshot_example",
        },
    },
]

_CONCEPTUAL_ONESHOT = {
    "mcq_data": {
        "question": (
            "Which of the following best describes the role of a Certificate Authority (CA) "
            "in a Public Key Infrastructure (PKI)?"
        ),
        "correct_answer": "It issues and signs digital certificates that bind a public key to an identity.",
        "rationale": (
            "A CA is a trusted third party that signs certificates, allowing others to verify "
            "that a given public key genuinely belongs to the claimed entity."
        ),
        "distractors": [
            {
                "option": "It encrypts data transmitted between two parties using symmetric keys.",
                "explanation": "Confuses the CA's role with the role of a session-key encryption layer.",
            },
            {
                "option": "It stores and distributes private keys on behalf of users.",
                "explanation": "Confuses certificate issuance with key escrow; a CA never holds user private keys.",
            },
            {
                "option": "It generates one-time passwords for multi-factor authentication.",
                "explanation": "Confuses PKI infrastructure with OTP/MFA mechanisms.",
            },
        ],
        "question_type": "conceptual",
        "source": "oneshot_example",
    },
}


def _sample_oneshot(topic: str, question_type: str) -> Dict:
    if question_type == "conceptual":
        return _CONCEPTUAL_ONESHOT
    t = topic.lower()
    if "sort" in t or "merge" in t:
        candidates = [_ONESHOT_POOL[1]]
    elif "bfs" in t or "graph" in t or "traversal" in t:
        candidates = [_ONESHOT_POOL[2]]
    elif "dynamic" in t or "dp" in t or "knapsack" in t:
        candidates = [_ONESHOT_POOL[3]]
    elif "bst" in t or "binary search tree" in t:
        candidates = [_ONESHOT_POOL[4]]
    else:
        candidates = _ONESHOT_POOL
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# Conceptual question quality guidance (FIX-I)
# ---------------------------------------------------------------------------

_CONCEPTUAL_QUALITY_GUIDE = """
=== CONCEPTUAL QUESTION QUALITY STANDARD ===
BAD — never generate these:
  ✗ "What is X?"            — pure definition, no thinking required
  ✗ "Define the term X."    — encyclopaedia question
  ✗ "Which is a property of X?" (where only one textbook fact is needed)

GOOD — use one of these styles:
  ✓ SCENARIO JUDGMENT: Give a concrete real-world situation, ask which approach is better or what will happen.
     Example: "A startup stores user passwords using unsalted MD5. During a breach, what is the primary risk compared to bcrypt?"
  ✓ CONCEPT DISCRIMINATION: Highlight the key difference between two easily confused concepts.
     Example: "A developer needs guaranteed delivery with ordered packets. Should they use TCP or UDP, and why?"
  ✓ FAULT DIAGNOSIS: Describe a system exhibiting a symptom, ask for root cause.
     Example: "A machine learning model achieves 99% training accuracy but 60% test accuracy. What is the most likely cause?"
  ✓ CROSS-CONCEPT REASONING: Require connecting ≥ 2 concepts to answer.
     Example: "If a distributed cache uses consistent hashing and one node fails, how does this affect cache hit rate and data redistribution compared to modulo hashing?"
"""


# ---------------------------------------------------------------------------
# Baseline Generator
# ---------------------------------------------------------------------------

class BaselineGenerator:
    """Control group: vector-retrieved context + one-shot prompt."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, topic: str, context: List[Dict]) -> Tuple[str, str]:
        context_str   = "\n".join([f"[{i+1}] {c['content'][:400]}" for i, c in enumerate(context)])
        question_type = _choose_question_type(topic)
        one_shot      = _sample_oneshot(topic, question_type)

        if question_type == "computational":
            prompt = f"""You are an elite Computer Science Professor and Psychometrician. Generate ONE highly discriminative computational MCQ based on the provided textbook context.

=== TEXTBOOK CONTEXT ===
{context_str}

=== DESIGN REQUIREMENTS (STRICT RUBRIC) ===
1. CROSS-CONCEPT SYNTHESIS (CRITICAL): Force integration of MULTIPLE DISTINCT logical components from the context.
2. EDGE-CASE TRIGGERING (CRITICAL): Do NOT use happy-path input data. Design inputs to force complex conditional branches.
3. SELF-CONTAINED TRACING: Include ALL required input data. Keep inputs small (array ≤ 6 elements).
4. RIGOROUS MATHEMATICAL TRACING: Trace the algorithm with absolute precision.

=== DISTRACTOR PROTOCOL ===
Generate exactly three distractors. Each MUST represent a specific, mathematically traceable cognitive error.
Tag each with: [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]

=== EXAMPLE ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "algorithm_rules": "Define the EXACT distinct concepts/formulas combined",
        "step_by_step_execution": "<Execute step-by-step, proving edge-case was triggered>",
        "final_state": "The absolute correct final output"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "...",
        "rationale": "...",
        "distractors": [
            {{"option": "...", "explanation": "[Error Tag] Detailed reason"}}
        ],
        "question_type": "computational",
        "source": "baseline_generated"
    }}
}}
"""
        else:
            prompt = f"""You are an elite Computer Science Professor. Generate ONE high-quality conceptual MCQ about "{topic}" based on the provided context.

=== TEXTBOOK CONTEXT ===
{context_str}

{_CONCEPTUAL_QUALITY_GUIDE}

=== EXAMPLE ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "core_concept": "The key concept being tested",
        "why_this_matters": "Why understanding this is important",
        "common_misconceptions": "List the misconceptions used as distractors"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "...",
        "rationale": "...",
        "distractors": [
            {{"option": "...", "explanation": "Why a student with misconception X would choose this"}}
        ],
        "question_type": "conceptual",
        "source": "baseline_generated"
    }}
}}
"""

        # ------------------------------------------------------------------
        # Difficulty-filter retry loop (VECTOR_RAG / BaselineGenerator)
        # boost_block starts empty; filled with judge feedback after EASY rating
        # ------------------------------------------------------------------
        # Difficulty-filter retry loop (VECTOR_RAG / BaselineGenerator)
        # Tracks the BEST candidate across all attempts (not just the last).
        # ------------------------------------------------------------------
        candidate_json = ""
        best_json      = ""
        best_score     = 0
        best_reasoning = ""
        last_score     = DIFFICULTY_THRESHOLD
        last_reasoning = ""
        boost_block    = ""

        for attempt in range(MAX_DIFFICULTY_RETRIES):
            active_prompt = boost_block + prompt
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": active_prompt}],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            try:
                data = json.loads(raw_content)
                if "mcq_data" in data:
                    flat_data = data["mcq_data"]
                    flat_data["generator_scratchpad"] = data.get("generator_scratchpad", {})
                    candidate_json = json.dumps(flat_data, ensure_ascii=False)
                else:
                    candidate_json = raw_content
            except Exception:
                candidate_json = raw_content

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(
                f"[DIFFICULTY FILTER | VECTOR_RAG | attempt {attempt+1}/{MAX_DIFFICULTY_RETRIES}] "
                f"Score={last_score}/5 — {last_reasoning}"
            )

            # Always keep the highest-scoring candidate seen so far
            if last_score > best_score:
                best_score     = last_score
                best_json      = candidate_json
                best_reasoning = last_reasoning

            if last_score > DIFFICULTY_THRESHOLD:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(
                f"  → Score {last_score}/5 ≤ threshold {DIFFICULTY_THRESHOLD}/5. "
                "Regenerating with difficulty boost …"
            )
        else:
            print(
                f"[DIFFICULTY FILTER | VECTOR_RAG] All {MAX_DIFFICULTY_RETRIES} attempts scored "
                f"≤ {DIFFICULTY_THRESHOLD}/5. Accepting best seen (score={best_score})."
            )

        # Use the best candidate across all attempts
        final_json      = best_json if best_json else candidate_json
        final_score     = best_score if best_json else last_score
        final_reasoning = best_reasoning if best_json else last_reasoning

        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = final_score
            q_data["difficulty_reasoning"] = final_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass

        return final_json, "baseline_generated"


# ---------------------------------------------------------------------------
# NoRetrievalGenerator
# ---------------------------------------------------------------------------

class NoRetrievalGenerator:
    """Pure LLM, zero retrieval context. True zero-retrieval baseline."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, topic: str) -> Tuple[str, str]:
        question_type = _choose_question_type(topic)
        one_shot      = _sample_oneshot(topic, question_type)

        if question_type == "computational":
            prompt = f"""You are an elite Computer Science Professor. Generate ONE highly discriminative computational MCQ about: "{topic}".

No external context. Use only your parametric knowledge.

=== DESIGN REQUIREMENTS ===
1. ONE ALGORITHM ONLY: Pick ONE algorithm related to "{topic}".
2. EDGE-CASE INPUT (CRITICAL): The edge case must come from the INPUT DATA:
   - Hash table: keys causing a maximum-length collision chain
   - Sorting: nearly-sorted, reverse-sorted, or all-equal input
   - Tree: degenerate insertion order (linear chain)
   - Graph: disconnected components or isolated nodes
   - DP: inputs forcing the non-trivial subproblem branch
3. SELF-CONTAINED: Include ALL input data. Arrays ≤ 6 elements, numbers ≤ 50.
4. MATHEMATICAL PRECISION: Compute answer step-by-step in generator_scratchpad FIRST.
5. DISTRACTORS: Each must use an explicit error tag:
   [Boundary Error] / [Initialization Error] / [Procedural Omission] / [Operator Confusion]

=== EXAMPLE FORMAT ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{{{
    "generator_scratchpad": {{{{
        "algorithm_rules": "Exact rules of the algorithm",
        "step_by_step_execution": "Step 1: ... Step 2: ... Final: ...",
        "final_state": "The exact correct answer"
    }}}},
    "mcq_data": {{{{
        "question": "...",
        "correct_answer": "...",
        "rationale": "Clear step-by-step explanation",
        "distractors": [
            {{{{"option": "...", "explanation": "[Error Tag] Why a student makes this mistake"}}}},
            {{{{"option": "...", "explanation": "[Error Tag] Why a student makes this mistake"}}}},
            {{{{"option": "...", "explanation": "[Error Tag] Why a student makes this mistake"}}}}
        ],
        "question_type": "computational",
        "source": "no_retrieval_generated"
    }}}}
}}}}"""
        else:
            prompt = f"""You are an elite Computer Science Professor. Generate ONE high-quality conceptual MCQ about: "{topic}".

No external context. Use only your parametric knowledge.

{_CONCEPTUAL_QUALITY_GUIDE}

=== EXAMPLE FORMAT ===
{json.dumps(one_shot, indent=2)}

=== REQUIRED JSON SCHEMA ===
{{{{
    "generator_scratchpad": {{{{
        "core_concept": "Key concept being tested",
        "cross_concept_link": "How two concepts connect",
        "common_misconceptions": "Misconceptions targeted by distractors"
    }}}},
    "mcq_data": {{{{
        "question": "...",
        "correct_answer": "...",
        "rationale": "...",
        "distractors": [
            {{{{"option": "...", "explanation": "Misconception: ..."}}}},
            {{{{"option": "...", "explanation": "Misconception: ..."}}}},
            {{{{"option": "...", "explanation": "Misconception: ..."}}}}
        ],
        "question_type": "conceptual",
        "source": "no_retrieval_generated"
    }}}}
}}}}"""

        # ------------------------------------------------------------------
        # Difficulty-filter retry loop (NO_RETRIEVAL)
        # Tracks the BEST candidate across all attempts (not just the last).
        # ------------------------------------------------------------------
        candidate_json = ""
        best_json      = ""
        best_score     = 0
        best_reasoning = ""
        last_score     = DIFFICULTY_THRESHOLD
        last_reasoning = ""
        boost_block    = ""

        for attempt in range(MAX_DIFFICULTY_RETRIES):
            active_prompt = boost_block + prompt
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": active_prompt}],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            try:
                data = json.loads(raw_content)
                if "mcq_data" in data:
                    flat = data["mcq_data"]
                    flat["generator_scratchpad"] = data.get("generator_scratchpad", {})
                    candidate_json = json.dumps(flat, ensure_ascii=False)
                else:
                    candidate_json = raw_content
            except Exception:
                candidate_json = raw_content

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(
                f"[DIFFICULTY FILTER | NO_RETRIEVAL | attempt {attempt+1}/{MAX_DIFFICULTY_RETRIES}] "
                f"Score={last_score}/5 — {last_reasoning}"
            )

            if last_score > best_score:
                best_score     = last_score
                best_json      = candidate_json
                best_reasoning = last_reasoning

            if last_score > DIFFICULTY_THRESHOLD:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(
                f"  → Score {last_score}/5 ≤ threshold {DIFFICULTY_THRESHOLD}/5. "
                "Regenerating with difficulty boost …"
            )
        else:
            print(
                f"[DIFFICULTY FILTER | NO_RETRIEVAL] All {MAX_DIFFICULTY_RETRIES} attempts scored "
                f"≤ {DIFFICULTY_THRESHOLD}/5. Accepting best seen (score={best_score})."
            )

        final_json      = best_json if best_json else candidate_json
        final_score     = best_score if best_json else last_score
        final_reasoning = best_reasoning if best_json else last_reasoning

        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = final_score
            q_data["difficulty_reasoning"] = final_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass

        return final_json, "no_retrieval"


# ---------------------------------------------------------------------------
# SmartGenerator (GraphRAG model)
# ---------------------------------------------------------------------------

class SmartGenerator:
    REUSE_THRESHOLD      = 0.90
    VARY_THRESHOLD_COMP  = 0.70
    VARY_THRESHOLD_CONC  = 0.60

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        topic: str,
        graph_context: Dict,
        qb_retriever,
    ) -> Tuple[str, str]:

        question_type  = _choose_question_type(topic)
        vary_threshold = (
            self.VARY_THRESHOLD_COMP if question_type == "computational"
            else self.VARY_THRESHOLD_CONC
        )

        if qb_retriever is not None and qb_retriever.available:
            best_q, best_sim = self._find_best_question(topic, question_type, qb_retriever)
            print(f"[DEBUG] topic='{topic}' | type='{question_type}'")
            print(f"[DEBUG] best_sim={best_sim:.4f} (REUSE>{self.REUSE_THRESHOLD}, VARY>{vary_threshold})")
            if best_q:
                print(f"[DEBUG] best_q={best_q.get('question','')[:80]}...")
            print()

            if best_q is not None:
                if best_sim >= self.REUSE_THRESHOLD:
                    raw = self._wrap_original(best_q, graph_context, question_type)
                    return self._apply_difficulty_filter(raw, "reused", topic, graph_context,
                                                         question_type, qb_retriever)
                if best_sim >= vary_threshold:
                    raw = self._vary(best_q, graph_context, question_type)
                    return self._apply_difficulty_filter(raw, "varied", topic, graph_context,
                                                         question_type, qb_retriever)

        few_shot_examples = []
        if qb_retriever is not None and qb_retriever.available:
            few_shot_examples = qb_retriever.retrieve_similar(
                topic,
                top_k=3,
                prefer_computational=(question_type == "computational"),
            )

        if question_type == "computational":
            raw = self._generate_fresh(topic, graph_context, few_shot_examples)
            return self._apply_difficulty_filter(raw, "generated", topic, graph_context,
                                                  question_type, qb_retriever)

    # ------------------------------------------------------------------
    # Difficulty-filter dispatcher for SmartGenerator (GRAPH_RAG)
    # ------------------------------------------------------------------

    def _apply_difficulty_filter(
        self,
        initial_json: str,
        initial_method: str,
        topic: str,
        graph_context: Dict,
        question_type: str,
        qb_retriever,
    ) -> Tuple[str, str]:
        """
        Evaluate the generated question's difficulty score (1-5).
        If score <= DIFFICULTY_THRESHOLD, regenerate with targeted boost
        (up to MAX_DIFFICULTY_RETRIES total attempts, counting the first).

        Returns (question_json, method_string).
        """
        candidate_json = initial_json
        final_method   = initial_method
        best_json      = ""
        best_score     = 0
        best_reasoning = ""
        last_score     = DIFFICULTY_THRESHOLD
        last_reasoning = ""

        # Attempt 1: the question was already generated before calling this method
        last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
        print(
            f"[DIFFICULTY FILTER | GRAPH_RAG | attempt 1/{MAX_DIFFICULTY_RETRIES}] "
            f"Score={last_score}/5 — {last_reasoning}"
        )
        if last_score > best_score:
            best_score     = last_score
            best_json      = candidate_json
            best_reasoning = last_reasoning

        for attempt in range(2, MAX_DIFFICULTY_RETRIES + 1):
            if last_score > DIFFICULTY_THRESHOLD:
                break

            boost_block = _build_difficulty_boost(candidate_json, last_score, last_reasoning)
            print(
                f"  → Score {last_score}/5 ≤ threshold {DIFFICULTY_THRESHOLD}/5. "
                f"Regenerating with difficulty boost (attempt {attempt}) …"
            )

            few_shot_examples = []
            if qb_retriever is not None and qb_retriever.available:
                few_shot_examples = qb_retriever.retrieve_similar(
                    topic, top_k=3, prefer_computational=(question_type == "computational")
                )

            if question_type == "computational":
                candidate_json = self._generate_fresh(topic, graph_context, few_shot_examples,
                                                       boost_block=boost_block)
                final_method   = "generated_difficulty_retry"
            else:
                candidate_json = self._generate_conceptual(topic, graph_context, few_shot_examples,
                                                            boost_block=boost_block)
                final_method   = "generated_difficulty_retry"

            last_score, last_reasoning = assess_difficulty(self.llm, candidate_json)
            print(
                f"[DIFFICULTY FILTER | GRAPH_RAG | attempt {attempt}/{MAX_DIFFICULTY_RETRIES}] "
                f"Score={last_score}/5 — {last_reasoning}"
            )

            if last_score > best_score:
                best_score     = last_score
                best_json      = candidate_json
                best_reasoning = last_reasoning
        else:
            if last_score <= DIFFICULTY_THRESHOLD:
                print(
                    f"[DIFFICULTY FILTER | GRAPH_RAG] All {MAX_DIFFICULTY_RETRIES} attempts scored "
                    f"≤ {DIFFICULTY_THRESHOLD}/5. Accepting best seen (score={best_score})."
                )

        final_json      = best_json if best_json else candidate_json
        final_score     = best_score if best_json else last_score
        final_reasoning = best_reasoning if best_json else last_reasoning

        try:
            q_data = json.loads(final_json)
            q_data["difficulty_score"]     = final_score
            q_data["difficulty_reasoning"] = final_reasoning
            final_json = json.dumps(q_data, ensure_ascii=False)
        except Exception:
            pass

        return final_json, final_method

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_query(self, topic: str, question_type: str) -> str:
        if question_type == "conceptual":
            return f"explain concept definition principle {topic} why how compare"
        templates = {
            "hash table":          f"load factor clustering worst-case probes deletion tombstone performance degradation {topic}",
            "sorting":             f"trace sort step by step array state after each pass swap comparison {topic}",
            "graph traversal":     f"BFS DFS trace visit order disconnected components isolated vertex distance {topic}",
            "dynamic programming": f"fill DP table step by step compute optimal subproblem value {topic}",
            "recursion":           f"trace recursive calls stack frames compute return value base case {topic}",
            "binary search tree":  f"trace BST insert delete search compare keys height in-order {topic}",
            "binary search":       f"trace binary search midpoint computation sorted array {topic}",
            "heap":                f"trace heap insert delete heapify parent child index {topic}",
            "graph":               f"trace graph algorithm adjacency matrix edge weight path {topic}",
        }
        for key, tmpl in templates.items():
            if key in topic.lower():
                return tmpl
        return f"trace algorithm step by step compute result concrete input {topic}"

    def _cosine_from_ip(self, ip_score: float) -> float:
        """IndexFlatIP on normalised vectors returns cosine directly."""
        return max(0.0, min(1.0, float(ip_score)))

    def _find_best_question(
        self, topic: str, question_type: str, qb_retriever
    ) -> Tuple[Optional[Dict], float]:
        """
        FIX-C + FIX-J: search for question_type-appropriate questions.
        Handles both legacy 'type' field and new 'question_type' field.
        Falls back to any type if none of the right type are found.
        """
        query     = self._build_query(topic, question_type)
        query_vec = qb_retriever.encoder.encode([query], convert_to_numpy=True)
        import faiss as _faiss
        _faiss.normalize_L2(query_vec)
        k         = min(50, len(qb_retriever.questions))
        scores, indices = qb_retriever.index.search(query_vec, k)

        best_q, best_sim = None, 0.0
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            q = qb_retriever.questions[idx]
            # FIX-J: handle both old 'type' and new 'question_type' fields
            q_type = q.get("question_type") or q.get("type", "")
            if q_type == question_type:
                sim = self._cosine_from_ip(score)
                if sim > best_sim:
                    best_sim = sim
                    best_q   = q

        return best_q, best_sim

    # ------------------------------------------------------------------
    # _wrap_original
    # ------------------------------------------------------------------

    def _wrap_original(self, source_q: Dict, graph_context: Dict, question_type: str) -> str:
        question  = source_q.get("question", "")
        answer    = source_q.get("answer", "")
        rationale = source_q.get("rationale", "")
        options   = source_q.get("options", [])

        raw_distractors = [
            opt for opt in options
            if answer not in opt and opt.strip() != answer.strip()
        ][:3]

        if question_type == "computational":
            prompt = f"""You are a CS exam quality assurance expert. Given an existing computational question and its answer, do two things:

1. Write a step-by-step scratchpad showing how to solve the problem mathematically.
2. Rewrite the distractors with explicit error tags and explanations.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{answer}

=== RATIONALE (for reference) ===
{rationale[:500]}

=== ORIGINAL DISTRACTORS ===
{json.dumps(raw_distractors)}

Return JSON ONLY:
{{
    "generator_scratchpad": {{
        "algorithm_rules": "Key algorithm rules being tested",
        "step_by_step_execution": "Step 1: ... Step 2: ... Step 3: ...",
        "final_state": "{answer}"
    }},
    "distractors": [
        {{"option": "distractor value", "explanation": "[Error Tag] Why a student makes this specific mistake"}},
        {{"option": "distractor value", "explanation": "[Error Tag] Why a student makes this specific mistake"}},
        {{"option": "distractor value", "explanation": "[Error Tag] Why a student makes this specific mistake"}}
    ]
}}

Error tags must be one of: [Boundary Error] [Initialization Error] [Procedural Omission] [Operator Confusion]"""
        else:
            prompt = f"""You are a CS exam quality assurance expert. Enrich this conceptual question's distractors with explicit misconception explanations.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{answer}

=== RATIONALE ===
{rationale[:500]}

=== ORIGINAL DISTRACTORS ===
{json.dumps(raw_distractors)}

Return JSON ONLY:
{{
    "generator_scratchpad": {{
        "core_concept": "What conceptual understanding this tests",
        "why_correct": "Why the correct answer is right",
        "common_misconceptions": "What misconceptions the distractors exploit"
    }},
    "distractors": [
        {{"option": "distractor text", "explanation": "Why a student with misconception X would choose this"}},
        {{"option": "distractor text", "explanation": "Why a student with misconception Y would choose this"}},
        {{"option": "distractor text", "explanation": "Why a student with misconception Z would choose this"}}
    ]
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1200,
            )
            enriched = json.loads(response.choices[0].message.content)
        except Exception:
            enriched = {
                "generator_scratchpad": {
                    "algorithm_rules": "",
                    "step_by_step_execution": rationale,
                    "final_state": answer,
                },
                "distractors": [
                    {"option": d, "explanation": "[Procedural Omission] Common mistake"}
                    for d in raw_distractors
                ],
            }

        result = {
            "question":             question,
            "correct_answer":       answer,
            "rationale":            rationale,
            "distractors":          enriched.get("distractors", []),
            "generator_scratchpad": enriched.get("generator_scratchpad", {}),
            "question_type":        question_type,
            "source":               "reused_from_bank",
        }
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # _vary
    # ------------------------------------------------------------------

    def _vary(self, source_q: Dict, context: Dict, question_type: str) -> str:
        for _ in range(3):
            raw = self._do_vary(source_q, context, question_type)
            if question_type == "conceptual" or self._verify_answer(raw):
                return raw
        return raw

    def _do_vary(self, source_q: Dict, context: Dict, question_type: str) -> str:
        context_str = "\n".join(
            [c["content"][:300] for c in context.get("nodes", [])[:3]]
        )
        relations = context.get("relations", [])[:8]
        if relations:
            rel_str = "\n".join(
                f"  ({e['subject']}) --[{e['predicate']}]--> ({e['object']})"
                for e in relations
            )
            graph_hint = f"""
    === KNOWLEDGE GRAPH RELATIONS — PRIMARY CONSTRAINT ===
    {rel_str}

    Scoring rule you MUST satisfy:
    - Chain >= 2 of the above relations into the question logic  →  graph_relational_depth 4-5
    - Only swap numbers without using any relation             →  graph_relational_depth 1 (FAIL)
    """
        else:
            graph_hint = ""

        if question_type == "computational":
            prompt = f"""You are a CS Professor creating a new exam question by modifying an existing one.

    === ORIGINAL QUESTION (use as structural reference only) ===
    Question:  {source_q['question']}
    Answer:    {source_q.get('answer', 'N/A')}
    Rationale: {source_q.get('rationale', 'N/A')[:400]}

    === SUPPORTING CONTEXT ===
    {context_str}
    {graph_hint}
    === YOUR TASK — follow these steps in order ===
    Step 1 — Select two relations from the graph above. Write them in
            generator_scratchpad.graph_concepts_used and explain how they chain together.
    Step 2 — Design an adversarial input that forces the algorithm into a worst-case or
            boundary condition (NOT a happy-path trace). Write this in edge_case_justification.
    Step 3 — Adjust the numerical values / array contents from the original question
            so they trigger the boundary condition from Step 2 AND embed the
            two graph relations from Step 1 into the question logic.
    Step 4 — Compute the correct answer step-by-step in step_by_step_execution,
            then copy the final value into mcq_data.correct_answer.

    Additional rules:
    - Arrays <= 6 elements, numbers <= 50
    - FORBIDDEN: do not generate a basic single-step trace question
    (e.g. "insert N keys and find where key X lands" with no further reasoning)
    - Every distractor MUST use one of these tags:
    [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]
    - Avoid undefined behaviour; state every tie-breaking rule explicitly

    === REQUIRED JSON SCHEMA ===
    {{
        "generator_scratchpad": {{
            "graph_concepts_used": "Which two relations you chained and why",
            "edge_case_justification": "Why this input is adversarial / boundary-triggering",
            "algorithm_rules": "Key rules of the algorithm being tested",
            "step_by_step_execution": "Step 1: ... Step 2: ... Final answer: ...",
            "final_state": "The exact correct answer"
        }},
        "mcq_data": {{
            "question": "New question that requires tracing through >= 2 graph relations",
            "correct_answer": "...",
            "distractors": [
                {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
                {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
                {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}}
            ],
            "rationale": "Step 1: ... Step 2: ... Step 3: ...",
            "question_type": "computational",
            "source": "varied_from_bank"
        }}
    }}"""
        else:
            prompt = f"""You are a CS Professor creating a new conceptual question by modifying an existing one.

=== ORIGINAL QUESTION ===
Question:  {source_q['question']}
Answer:    {source_q.get('answer', 'N/A')}
Rationale: {source_q.get('rationale', 'N/A')[:400]}

=== SUPPORTING CONTEXT ===
{context_str}
{graph_hint}
{_CONCEPTUAL_QUALITY_GUIDE}

=== YOUR TASK ===
Create a NEW question that tests the SAME conceptual understanding but from a different angle.
Change the scenario, the framing, or the specific aspect being tested.
If graph relations are provided, use them to create a cross-concept question requiring causal reasoning.

=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "core_concept": "What this question tests",
        "variation_strategy": "How you changed the angle of the original",
        "common_misconceptions": "Misconceptions targeted by distractors"
    }},
    "mcq_data": {{
        "question": "...new question with different framing...",
        "correct_answer": "...",
        "distractors": [
            {{"option": "...", "explanation": "Why a student holding misconception X would choose this"}},
            {{"option": "...", "explanation": "Why a student holding misconception Y would choose this"}},
            {{"option": "...", "explanation": "Why a student holding misconception Z would choose this"}}
        ],
        "rationale": "Clear explanation of why the answer is correct.",
        "question_type": "conceptual",
        "source": "varied_from_bank"
    }}
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1800,
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            if "mcq_data" in data:
                flat = data["mcq_data"]
                flat["generator_scratchpad"] = data.get("generator_scratchpad", {})
                return json.dumps(flat, ensure_ascii=False)
            return raw
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # FIX-G: RV-Bench generation now uses graph relations for distractors
    # ------------------------------------------------------------------

    def _generate_rv_question(
        self,
        topic: str,
        graph_context: Optional[Dict] = None,  # FIX-G
    ) -> Optional[str]:
        """
        1. Call rv_generator to get question + code-verified answer.
        2. Call LLM to generate distractors.
           FIX-G: if graph_context has relations, inject them into the
           distractor prompt so the LLM can create distractors that reference
           causally related concepts (improving multi_hop_dependency score).
        3. Return final MCQ JSON or None if no RV class exists.
        """
        rv = generate_rv_question(topic)
        if rv is None:
            return None

        question  = rv["question"]
        answer    = rv["answer"]
        algorithm = rv["algorithm"]
        meta      = rv.get("meta", {})

        # FIX-G: build graph relations context for distractor enrichment
        graph_relations_block = ""
        if graph_context and graph_context.get("relations"):
            relations = graph_context["relations"][:10]
            rel_lines = "\n".join(
                f"  ({e['subject']}) --[{e['predicate']}]--> ({e['object']})"
                for e in relations
            )
            graph_relations_block = f"""
=== KNOWLEDGE GRAPH RELATIONS (use these to create richer distractors) ===
{rel_lines}

When designing distractors, try to reference the causal dependencies shown above.
For example, a distractor might arise from a student misapplying a relation
(e.g. using the wrong formula from the graph, or applying a step out of order).
This makes the question test multi-hop reasoning, not just a single rule.
"""

        distractor_prompt = f"""You are a CS exam designer. You have a verified computational question and its CORRECT answer (computed by Python code — do NOT change it).

Your job: generate exactly 3 WRONG answer options (distractors) that represent specific, common student mistakes.

=== QUESTION ===
{question}

=== CORRECT ANSWER (verified by code) ===
{answer}

=== ALGORITHM ===
{algorithm}

=== CONTEXT (algorithm parameters) ===
{json.dumps(meta, indent=2)}
{graph_relations_block}
=== DISTRACTOR RULES ===
Each distractor must:
1. Be a plausible WRONG answer a student might compute by making ONE specific mistake.
2. Have an explicit error tag: [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]
3. Be clearly different from the correct answer ({answer}).
4. Be a concrete value (not "it depends" or a range).

=== REQUIRED JSON (return ONLY this, no markdown) ===
{{
    "rationale": "Step-by-step explanation of how to reach the correct answer: {answer}",
    "distractors": [
        {{"option": "<wrong answer 1>", "explanation": "[Error Tag] Specific traceable mistake"}},
        {{"option": "<wrong answer 2>", "explanation": "[Error Tag] Specific traceable mistake"}},
        {{"option": "<wrong answer 3>", "explanation": "[Error Tag] Specific traceable mistake"}}
    ]
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": distractor_prompt}],
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            llm_out     = json.loads(response.choices[0].message.content)
            distractors = llm_out.get("distractors", [])
            rationale   = llm_out.get("rationale", "")
        except Exception:
            distractors = [
                {"option": "unknown", "explanation": "[Procedural Omission] Could not generate distractor"},
            ]
            rationale = ""

        result = {
            "question":             question,
            "correct_answer":       answer,
            "rationale":            rationale,
            "distractors":          distractors,
            "question_type":        "computational",
            "source":               "rv_generated",
            "generator_scratchpad": {
                "algorithm_rules":        f"RV-Bench: {algorithm}",
                "step_by_step_execution": rationale,
                "final_state":            answer,
                "rv_meta":                meta,
            },
        }
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # FIX-H: _generate_fresh NOW uses graph relations
    # ------------------------------------------------------------------

    def _generate_fresh(self, topic: str, context: Dict, few_shot_examples: List[Dict],
                        boost_block: str = "") -> str:
        for _ in range(3):
            raw = self._do_generate_fresh(topic, context, few_shot_examples, boost_block)
            if self._verify_answer(raw):
                return raw
        return raw

    def _do_generate_fresh(self, topic: str, context: Dict, few_shot_examples: List[Dict],
                           boost_block: str = "") -> str:
        one_shot = _sample_oneshot(topic, "computational")
        fewshot_block = "=== EXAMPLE FORMAT ===\n" + json.dumps(one_shot, indent=2) + "\n\n"
        if few_shot_examples:
            fewshot_block += "=== ADDITIONAL REFERENCE EXAMPLES ===\n" + "\n".join(
                [f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples)]
            ) + "\n\n"

        # FIX-H: build graph relations block
        relations = context.get("relations", [])
        nodes     = context.get("nodes", [])

        graph_block = ""
        if relations:
            rel_lines = "\n".join(
                f"  ({e['subject']}) --[{e['predicate']}]--> ({e['object']})"
                for e in relations[:15]
            )
            graph_block = f"""
=== KNOWLEDGE GRAPH RELATIONS FOR "{topic}" ===
{rel_lines}

=== MANDATORY GRAPH USAGE INSTRUCTION ===
You MUST generate a question that requires understanding AT LEAST TWO of the
above relations in sequence to answer correctly.

Example of how to use the graph:
If the graph shows:
  (hash function h(k)) --[PRODUCES_OUTPUT]--> (initial slot index)
  (initial slot index) --[HAS_STEP]--> (linear probe on collision)
Then design a question where the student must:
  Step 1: Compute the initial slot from h(k)
  Step 2: Apply the collision resolution rule to get the FINAL position
This forces multi-hop reasoning that a student who only knows one step cannot answer.

If you do NOT use the graph relations, you are producing a simple single-concept
question that graph retrieval adds no value to — please use the relations.
"""
        elif nodes:
            node_str = "\n".join([f"  [{i+1}] {n['content'][:200]}" for i, n in enumerate(nodes[:5])])
            graph_block = f"\n=== RETRIEVED CONTEXT ===\n{node_str}\n"

        prompt = f"""{boost_block}You are an elite Computer Science Professor and Psychometrician specialising in adversarial exam design.

Generate ONE highly discriminative computational MCQ about the topic: "{topic}".
{graph_block}
=== CRITICAL GENERATION CONSTRAINTS (MANDATORY) ===
You are strictly FORBIDDEN from generating basic "Trace the algorithm" questions (e.g., "Trace fib(5)", "What is the output of this sort?", "Insert these elements into a BST"). 

To receive a passing grade, your question MUST:
1. CROSS-CONCEPT SYNTHESIS (Multi-Hop): You MUST combine at least TWO distinct concepts from the provided Graph Relations. (e.g., How does choosing a specific Hash Table collision resolution strategy affect the worst-case BFS queue size? Or how does recursive call stack depth translate to DP table dimensions?)
2. EDGE CASE TRIGGERING: Do not use "happy path" inputs. You must construct an adversarial input scenario (e.g., a degenerate graph, extreme hash collisions, reverse-sorted arrays) that forces the algorithm into its worst-case or boundary condition.
3. CAUSAL REASONING: The question must ask the student to identify *why* a system failed, *how* two mechanisms interact, or compare the *efficiency trade-offs* of two approaches, rather than just computing a final single number.

If you generate a simple step-by-step trace question, you will fail.

5. DISTRACTOR QUALITY: Every distractor must name a specific, traceable student error with an explicit tag:
   [Boundary Error] | [Initialization Error] | [Procedural Omission] | [Operator Confusion]
6. AVOID UNDEFINED BEHAVIOUR: State every tie-breaking rule.

{fewshot_block}=== REQUIRED JSON SCHEMA ===
Complete "generator_scratchpad" FIRST, then fill "mcq_data".
{{
    "generator_scratchpad": {{
        "chosen_algorithm": "Name the primary algorithms interacting",
        "graph_concepts_used": "Which graph relations you used to create multi-hop depth",
        "edge_case_justification": "Explain WHY this input is an edge case",
        "step_by_step_execution": "Trace every step with exact values",
        "final_state": "The exact correct answer"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "<must match final_state exactly>",
        "rationale": "Clear step-by-step explanation matching the trace above",
        "distractors": [
            {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
            {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}},
            {{"option": "...", "explanation": "[Error Tag] Specific traceable mistake"}}
        ],
        "question_type": "computational",
        "source": "generated_via_graph"
    }}
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
            if "mcq_data" in data:
                flat_data = data["mcq_data"]
                flat_data["generator_scratchpad"] = data.get("generator_scratchpad", {})
                return json.dumps(flat_data, ensure_ascii=False)
            return raw_content
        except Exception:
            return raw_content

    # ------------------------------------------------------------------
    # FIX-I: _generate_conceptual with good/bad question guidance
    # ------------------------------------------------------------------

    def _generate_conceptual(
        self, topic: str, context: Dict, few_shot_examples: List[Dict],
        boost_block: str = ""
    ) -> str:
        nodes_str = "\n".join(
            [f"[{i+1}] {n['content'][:400]}" for i, n in enumerate(context.get("nodes", []))]
        )

        # Also inject graph relations for conceptual topics
        relations = context.get("relations", [])[:10]
        if relations:
            rel_str = "\n".join(
                f"  ({e['subject']}) --[{e['predicate']}]--> ({e['object']})"
                for e in relations
            )
            graph_block = f"\n=== KNOWLEDGE GRAPH RELATIONS ===\n{rel_str}\nUse these causal links to create cross-concept questions.\n"
        else:
            graph_block = ""

        one_shot = _sample_oneshot(topic, "conceptual")
        fewshot_block = "=== EXAMPLE FORMAT ===\n" + json.dumps(one_shot, indent=2) + "\n\n"
        if few_shot_examples:
            fewshot_block += "=== ADDITIONAL REFERENCE EXAMPLES ===\n" + "\n".join(
                [f"Ex {i+1}: {json.dumps(ex)}" for i, ex in enumerate(few_shot_examples)]
            ) + "\n\n"

        prompt = f"""{boost_block}You are an elite Computer Science Professor. Generate ONE high-quality conceptual MCQ about "{topic}".

=== RETRIEVED CONTEXT ===
{nodes_str}
{graph_block}
{_CONCEPTUAL_QUALITY_GUIDE}

=== DESIGN REQUIREMENTS ===
1. DEEP UNDERSTANDING: Test WHY and HOW, not just WHAT.
2. MISCONCEPTION-DRIVEN DISTRACTORS: Each wrong option must represent a specific, named misconception.
3. CROSS-CONCEPT: The question should require connecting at least two related concepts.
4. RELEVANCE: The question must be explicitly about "{topic}".

{fewshot_block}=== REQUIRED JSON SCHEMA ===
{{
    "generator_scratchpad": {{
        "core_concept": "The key concept being tested",
        "cross_concept_link": "How two or more concepts are connected in this question",
        "common_misconceptions": "The misconceptions each distractor exploits"
    }},
    "mcq_data": {{
        "question": "...",
        "correct_answer": "...",
        "rationale": "Clear explanation of why the answer is correct and why distractors are wrong.",
        "distractors": [
            {{"option": "...", "explanation": "Misconception: ..."}},
            {{"option": "...", "explanation": "Misconception: ..."}},
            {{"option": "...", "explanation": "Misconception: ..."}}
        ],
        "question_type": "conceptual",
        "source": "generated_via_graph"
    }}
}}"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
            if "mcq_data" in data:
                flat_data = data["mcq_data"]
                flat_data["generator_scratchpad"] = data.get("generator_scratchpad", {})
                return json.dumps(flat_data, ensure_ascii=False)
            return raw_content
        except Exception:
            return raw_content

    # ------------------------------------------------------------------
    # _verify_answer (Fix-1 retained)
    # ------------------------------------------------------------------

    def _verify_answer(self, question_json_str: str) -> bool:
        try:
            q = json.loads(question_json_str)
        except Exception:
            return False

        if q.get("question_type") == "conceptual":
            return True

        verify_prompt = (
            f"Solve this CS problem independently, showing all work step by step.\n\n"
            f"{q.get('question', '')}\n\n"
            f"At the very end write ONE line:\n"
            f"MY_ANSWER: <your computed result>"
        )

        try:
            res = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0,
                max_tokens=800,
            )
            text  = res.choices[0].message.content
            match = re.search(r"MY_ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if not match:
                return True

            computed = re.sub(r"^[A-Da-d][.)]\s*", "", match.group(1).strip()).strip().lower()
            expected = re.sub(r"^[A-Da-d][.)]\s*", "", q.get("correct_answer", "").strip()).strip().lower()

            if computed == expected:
                return True
            if expected in computed or computed in expected:
                return True
            try:
                if abs(float(computed.split()[0]) - float(expected.split()[0])) < 0.01:
                    return True
            except (ValueError, TypeError, IndexError):
                pass
            return False

        except Exception:
            return True
