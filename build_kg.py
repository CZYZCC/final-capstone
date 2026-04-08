import json
import os
import time
import re
import argparse
from typing import List, Dict
import random
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从 .env 文件读取
API_KEY = os.getenv("DEEPSEEK_API_KEY")


# ==========================================
# 1. 从教材提取知识图谱三元组 (unchanged)
# ==========================================
class LLMGraphExtractor:
    def __init__(self, api_key: str, output_file: str):
        self.llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.output_file = output_file
        self.triplets = []
        self.processed_nodes = set()
        self._load_progress()

    def _load_progress(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.triplets = data.get("triplets", [])
                    self.processed_nodes = set(data.get("processed_nodes", []))
                print(f"[*] 已加载历史进度：{len(self.triplets)} 条关系，{len(self.processed_nodes)} 个节点。")
            except Exception as e:
                print(f"[!] 读取历史进度失败，将重新开始: {e}")

    def _save_progress(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "processed_nodes": list(self.processed_nodes),
                "triplets": self.triplets
            }, f, indent=2, ensure_ascii=False)

    def clean_json_string(self, raw_str: str) -> str:
        clean_str = re.sub(r'```json\n?', '', raw_str)
        clean_str = re.sub(r'```', '', clean_str)
        return clean_str.strip()

    def extract_from_text(self, node_id: str, text: str) -> List[Dict]:
        system_prompt = """You are a CS compiler expert building an Executable Algorithm Knowledge Graph (GEAKG).
Extract entities, relations, and Control Data Flow (CDFG) properties from the textbook.

PRIORITY: Extract actionable, parameterizable logic that enables computational question generation.

Valid relations:
- HAS_STEP: A sequential algorithmic step.
- TRUE_BRANCH / FALSE_BRANCH: Conditional execution paths.
- PRODUCES_OUTPUT: State transition or return value.
- HAS_COMPLEXITY: Time/space formulas.

Rules:
- DO NOT artificially truncate entities to 1-4 words. Preserve exact formulas (e.g., "h(k) = k mod m"), pseudocode snippets, and full condition clauses (e.g., "if array[i] <= pivot").
- Treat the algorithm as a State Machine.

Return ONLY valid JSON:
{"triplets": [{"head": "...", "relation": "...", "tail": "..."}]}
"""
        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text:\n{text[:1500]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_content = response.choices[0].message.content
            clean_content = self.clean_json_string(raw_content)
            data = json.loads(clean_content)
            extracted = data.get("triplets", [])
            for triplet in extracted:
                triplet['source_node'] = node_id
            return extracted
        except Exception as e:
            print(f"  [!] 节点 {node_id} 提取失败: {e}")
            return []

    def process_dataset(self, textbook_dir: str):
        print(f"\n{'='*50}\n开始扫描并抽取教材知识图谱\n{'='*50}")
        chunks_to_process = []
        for i in range(1, 21):
            tb_path = os.path.join(textbook_dir, f"textbook{i}", f"textbook{i}_structured.json")
            if not os.path.exists(tb_path):
                continue
            with open(tb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for j, item in enumerate(data):
                    node_id = f"tb{i}_node{j}"
                    content = str(item.get('content', str(item)))
                    if node_id not in self.processed_nodes and len(content) > 50:
                        chunks_to_process.append((node_id, content))

        print(f"[*] 共发现 {len(chunks_to_process)} 个待处理的新文本块。")
        for idx, (node_id, content) in enumerate(chunks_to_process, 1):
            print(f"[{idx}/{len(chunks_to_process)}] 正在抽取 {node_id} ...", end="", flush=True)
            new_triplets = self.extract_from_text(node_id, content)
            if new_triplets:
                self.triplets.extend(new_triplets)
                print(f" 成功提取 {len(new_triplets)} 条关系.")
            else:
                print(" 未发现有价值关系.")
            self.processed_nodes.add(node_id)
            if idx % 20 == 0:
                self._save_progress()
            time.sleep(0.5)

        self._save_progress()
        print(f"\n{'='*50}\n抽取完成！共获得 {len(self.triplets)} 条逻辑边。")


# ==========================================
# 2. 从数据集构建 Few-shot 题库 (unchanged)
# ==========================================
class QuestionBankBuilder:
    COMP_KEYWORDS = [
        'calculate', 'compute', 'trace', 'run through', 'what is the value',
        'how many', 'insert the following', 'delete', 'find the index',
        'what will', 'after applying', 'result of', 'fill in', 'given the array',
        'hash function', 'sort the following', 'apply the algorithm',
        'step by step', 'iteration', 'what is output', 'what does',
        'given the sequence', 'state of the', 'after inserting', 'after deleting',
        'worst case', 'best case', 'recurrence', 'show the'
    ]

    def __init__(self, api_key: str, output_file: str):
        self.llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.output_file = output_file
        self.questions: List[Dict] = []
        self._load_progress()

    def _load_progress(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    self.questions = json.load(f)
                comp = sum(1 for q in self.questions if q.get('type') == 'computational')
                print(f"[*] 已加载 {len(self.questions)} 道题（{comp} 计算 / {len(self.questions)-comp} 概念）")
            except Exception as e:
                print(f"[!] 读取题库失败: {e}")

    def _save(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.questions, f, indent=2, ensure_ascii=False)

    def _classify_question(self, question_text: str) -> str:
        q_lower = question_text.lower()
        keyword_hits = sum(1 for kw in self.COMP_KEYWORDS if kw in q_lower)
        if keyword_hits >= 2:
            return "computational"

        prompt = (
            "Classify this CS question as EXACTLY one of:\n"
            "- \"computational\": requires tracing an algorithm, calculating a value, "
            "filling a table, or computing a specific numeric/symbolic result\n"
            "- \"conceptual\": asks for definitions, explanations, comparisons, "
            "or high-level understanding\n\n"
            "Return JSON only: {\"type\": \"computational\"} or {\"type\": \"conceptual\"}\n\n"
            f"Question: {question_text[:600]}"
        )
        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("type", "conceptual")
        except Exception:
            return "conceptual"

    def _load_raw_questions(self, questions_dir: str) -> List[Dict]:
        raw = []
        if not os.path.exists(questions_dir):
            print(f"[!] 找不到题目目录: {questions_dir}")
            return raw

        for fname in sorted(os.listdir(questions_dir)):
            fpath = os.path.join(questions_dir, fname)
            if fname.endswith('.jsonl'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                raw.append(json.loads(line))
                            except Exception:
                                pass
            elif fname.endswith('.json'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            raw.extend(data)
                        elif isinstance(data, dict):
                            raw.extend(data['questions']) if 'questions' in data else raw.append(data)
                    except Exception as e:
                        print(f"  [!] 读取 {fname} 失败: {e}")

        print(f"[*] 从 {questions_dir} 共读取 {len(raw)} 道原始题目")
        return raw

    def _normalize_question(self, raw: Dict, idx: int) -> Dict:
        q_text = (raw.get('Question') or raw.get('question') or
                  raw.get('text') or raw.get('problem') or '')
        answer = (raw.get('Answer') or raw.get('answer') or
                  raw.get('correct_answer') or raw.get('gold_answer') or '')
        raw_choices = raw.get('Choices') or raw.get('choices') or raw.get('options') or {}
        if isinstance(raw_choices, dict):
            options = [f"{k}. {v}" for k, v in raw_choices.items()]
        elif isinstance(raw_choices, list):
            options = raw_choices
        else:
            options = []
        topic    = (raw.get('Level-1 Topic') or raw.get('topic') or
                    raw.get('subject') or raw.get('category') or '')
        subtopic = raw.get('Level-2 Topic') or raw.get('subtopic') or ''
        rationale = raw.get('Rationale') or raw.get('rationale') or ''

        return {
            'id':              raw.get('id') or raw.get('qid') or f'q_{idx:04d}',
            'question':        str(q_text).strip(),
            'answer':          str(answer).strip(),
            'options':         options,
            'topic':           str(topic).strip(),
            'subtopic':        str(subtopic).strip(),
            'rationale':       str(rationale).strip(),
            'source_textbook': str(raw.get('textbook') or raw.get('source') or '').strip(),
            'type':            None,
            'raw':             raw
        }

    def build(self, questions_dir: str):
        print(f"\n{'='*50}\n开始构建 Few-shot 题库\n{'='*50}")
        raw_questions = self._load_raw_questions(questions_dir)
        processed_ids = {q['id'] for q in self.questions}
        to_process = []
        for idx, raw in enumerate(raw_questions):
            norm = self._normalize_question(raw, idx)
            if norm['id'] not in processed_ids and len(norm['question']) > 20:
                to_process.append(norm)

        print(f"[*] 需要新分类的题目数：{len(to_process)}")
        if not to_process:
            print("[*] 所有题目均已分类，无需重复处理。")
            return

        for idx, q in enumerate(to_process, 1):
            q['type'] = self._classify_question(q['question'])
            self.questions.append(q)
            tag = "💻 计算" if q['type'] == 'computational' else "📖 概念"
            print(f"  [{idx:4d}/{len(to_process)}] {tag} | {q['question'][:70]}...")
            if idx % 50 == 0:
                self._save()
        self._save()
        comp = sum(1 for q in self.questions if q.get('type') == 'computational')
        print(f"\n题库构建完成！共 {len(self.questions)} 道，计算题 {comp} 道")


# ==========================================
# 3. NEW: 从 KG triplets 直接生成计算题
# ==========================================
class KGBasedQuestionBuilder:
    """
    Generates computational MCQs directly from the KG triplets, guaranteeing
    topical coverage for any algorithm you specify.
    """

    N_PER_TOPIC        = 10   # total questions to generate per topic
    TRIPLETS_PER_CALL  = 8   # triplets to include per single-question call

    TARGET_TOPICS = [
        {
            "name": "hash table linear probing",
            "keywords": [
                "hash", "linear prob", "h(k)", "collision", "open addressing",
                "load factor", "cluster", "primary cluster", "worst case",
                "deletion", "tombstone", "rehash", "probe sequence",
            ],
        },
        {
            "name": "hash table quadratic probing",
            "keywords": [
                "quadratic prob", "hash", "double hash", "secondary cluster",
                "load factor", "collision", "probe sequence", "worst case",
            ],
        },
        {
            "name": "recursion and recurrence",
            "keywords": ["recursion", "recurrence", "base case", "T(n)", "recursive"],
        },
        {
            "name": "merge sort",
            "keywords": ["merge sort", "mergesort", "divide and conquer sort"],
        },
        {
            "name": "quick sort",
            "keywords": ["quicksort", "quick sort", "pivot", "partition"],
        },
        {
            "name": "insertion sort",
            "keywords": ["insertion sort"],
        },
        {
            "name": "heap sort and binary heap",
            "keywords": ["heap", "heapify", "max-heap", "min-heap"],
        },
        {
            "name": "BFS graph traversal",
            "keywords": ["bfs", "breadth-first", "queue", "visited", "component"],
        },
        {
            "name": "DFS graph traversal",
            "keywords": ["dfs", "depth-first", "stack", "back edge", "topological"],
        },
        {
            "name": "dynamic programming knapsack",
            "keywords": ["dynamic programming", "dp table", "knapsack", "subproblem"],
        },
        {
            "name": "dynamic programming LCS",
            "keywords": ["longest common subsequence", "lcs", "edit distance"],
        },
        {
            "name": "binary search tree operations",
            "keywords": ["binary search tree", "bst", "inorder", "predecessor", "successor"],
        },
        {
            "name": "Dijkstra shortest path",
            "keywords": ["dijkstra", "shortest path", "relaxation", "priority queue"],
        },
    ]

    def __init__(self, api_key: str, kg_path: str, output_file: str):
        self.llm         = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.kg_path     = kg_path
        self.output_file = output_file
        self.questions: List[Dict] = []
        self._load_progress()

    def _load_progress(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    self.questions = json.load(f)
                kg_gen = sum(1 for q in self.questions if q.get('source') == 'kg_generated')
                print(f"[*] 已加载 {len(self.questions)} 道题（其中 {kg_gen} 道来自 KG 生成）")
            except Exception as e:
                print(f"[!] 读取题库失败: {e}")

    def _save(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.questions, f, indent=2, ensure_ascii=False)

    def _load_triplets(self) -> List[Dict]:
        with open(self.kg_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('triplets', [])

    def _filter_triplets(self, triplets: List[Dict], keywords: List[str]) -> List[Dict]:
        """Return triplets that contain any of the keywords."""
        result = []
        for t in triplets:
            combined = (t.get('head', '') + ' ' + t.get('tail', '')).lower()
            if any(kw.lower() in combined for kw in keywords):
                result.append(t)
        return result

    def _already_covered(self, topic_name: str) -> bool:
        """Return True if we already have ≥5 kg_generated questions for this topic."""
        existing = [
            q for q in self.questions
            if q.get('source') == 'kg_generated'
            and q.get('topic', '').lower() == topic_name.lower()
        ]
        return len(existing) >= self.N_PER_TOPIC

    def _generate_one(self, topic: Dict, triplet_batch: List[Dict]) -> Dict:
        """Generate exactly ONE computational question from a batch of triplets."""
        triplet_str = "\n".join(
            f"  ({t['head'][:60]}) --[{t['relation']}]--> ({t['tail'][:80]})"
            for t in triplet_batch
        )

        prompt = f"""You are a CS professor specialising in adversarial exam design. Generate ONE computational MCQ about "{topic['name']}".

=== KNOWLEDGE GRAPH RELATIONS (you MUST use at least TWO of these) ===
{triplet_str}

=== MANDATORY DEPTH REQUIREMENTS ===
You are FORBIDDEN from generating a basic "insert N keys, find where key X lands" question.
Such questions only test one step and score 1/5 on graph_relational_depth.

Your question MUST satisfy ALL of the following:
1. CHAIN TWO RELATIONS: The correct answer can only be found by following at least two
   of the above graph relations in sequence.
   Example chain: (h(k)=k mod m) --[PRODUCES_OUTPUT]--> (initial slot)
                  (initial slot) --[HAS_STEP]--> (linear probe on collision)
                  (linear probe) --[HAS_COMPLEXITY]--> (O(1+α) expected)
   → Question: "After inserting these keys, what is the load factor, and how many probes
     does the NEXT insertion require in the worst case?"

2. ADVERSARIAL INPUT: Construct input that forces the algorithm into a worst-case or
   boundary condition (maximum collision chain, table near-full, all keys same hash).

3. MULTI-CONCEPT ANSWER: The answer must combine results from two distinct steps
   (e.g., "load factor = X/N, therefore expected probes = 1/(1-α) = Y").

=== STRICT FORMAT RULES ===
1. Use TINY concrete inputs: table size ≤ 8, at most 5 keys, numbers ≤ 30
2. COMPUTE the answer step-by-step in "scratchpad" FIRST
3. "correct_answer" must be ONLY the final value — NO option letters.
   Write "3 probes" not "B. 3 probes".
4. Each option in "options" must be a full string like "A. 2 probes"
5. The correct answer must appear as one of the 4 options

=== EXAMPLE OF A GOOD QUESTION ===
{{
  "scratchpad": "Insert 10,15,20 into table size 5, h(k)=k mod 5. All hash to slot 0. Load factor = 3/5 = 0.6. Next insertion worst case probes = table_size - filled_slots_in_chain + 1 = need to probe slots 0(full),1(full),2(full),3(empty) = 3 probes. But also must count the initial probe, total = 4.",
  "question": "A hash table of size 5 uses linear probing with h(k) = k mod 5. After inserting keys [10, 15, 20], what is the load factor AND the worst-case number of probes needed to insert the next key that hashes to slot 0?",
  "correct_answer": "Load factor 0.6; 4 probes",
  "options": ["A. Load factor 0.6; 4 probes", "B. Load factor 0.6; 3 probes", "C. Load factor 3; 4 probes", "D. Load factor 0.4; 2 probes"],
  "rationale": "3 keys / 5 slots = 0.6. Slots 0,1,2 occupied. Next key hashing to 0 probes: slot 0 (full) → slot 1 (full) → slot 2 (full) → slot 3 (empty) = 4 probes total."
}}

Return a single JSON object with exactly these keys: "scratchpad", "question", "correct_answer", "options", "rationale"."""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1800,
                temperature=0.8,
            )
            raw  = response.choices[0].message.content
            data = json.loads(raw)
            if data.get("question") and data.get("correct_answer"):
                ans = data["correct_answer"].strip()
                ans = re.sub(r'^[A-Da-d][.\)]\s*', '', ans).strip()
                data["correct_answer"] = ans
                return data
            return {}
        except Exception as e:
            print(f"    [!] 单题生成失败: {e}")
            return {}

    def _verify_question(self, q_dict: Dict) -> bool:
        question_text = q_dict.get("question", "")
        stated_answer = q_dict.get("correct_answer", "")
        if not question_text or not stated_answer:
            return False

        verify_prompt = (
            f"Solve this CS problem independently, showing all work step by step.\n\n"
            f"{question_text}\n\n"
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
            text = res.choices[0].message.content

            import re
            match = re.search(r"MY_ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if not match:
                return True

            computed = match.group(1).strip().lower()
            computed = re.sub(r'^[a-d][.\)]\s*', '', computed).strip()
            expected = re.sub(r'^[a-d][.\)]\s*', '', stated_answer.strip().lower()).strip()

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

    def _to_bank_format(self, raw_q: Dict, topic_name: str, q_idx: int) -> Dict:
        return {
            "id":              f"kg_{topic_name.replace(' ', '_')}_{q_idx:03d}",
            "question":        raw_q.get("question", ""),
            "answer":          raw_q.get("correct_answer", ""),
            "options":         raw_q.get("options", []),
            "rationale":       raw_q.get("rationale", ""),
            "scratchpad":      raw_q.get("scratchpad", ""),
            "topic":           topic_name,
            "subtopic":        "",
            "source":          "kg_generated",
            "type":            "computational",
        }

    def build(self):
        """Main entry point.  Generates questions for all TARGET_TOPICS."""
        print(f"\n{'='*50}\n开始从 KG 生成计算题\n{'='*50}")

        if not os.path.exists(self.kg_path):
            print(f"[!] 找不到 KG 文件: {self.kg_path}，跳过。")
            return

        all_triplets = self._load_triplets()
        print(f"[*] 已加载 {len(all_triplets)} 条 triplet")

        total_new = 0

        for topic in self.TARGET_TOPICS:
            tname = topic["name"]

            # 动态计算当前还需要生成几道题
            existing_count = sum(1 for q in self.questions if q.get('source') == 'kg_generated' and q.get('topic', '').lower() == tname.lower())
            needed = self.N_PER_TOPIC - existing_count

            if needed <= 0:
                print(f"  [=] '{tname}' 已有足够题目，跳过。")
                continue

            matching = self._filter_triplets(all_triplets, topic["keywords"])
            if not matching:
                print(f"  [!] '{tname}' 未找到相关 triplet，跳过。")
                continue

            print(f"\n  >>> 话题: {tname}")
            print(f"      找到 {len(matching)} 条相关 triplet，尚需补充 {needed} 道题")

            import random
            random.shuffle(matching)

            topic_count = 0
            attempts = 0
            max_attempts = needed * 3  # 防止生成一直失败导致死循环

            # 改用 while 循环，缺几道就补几道
            while topic_count < needed and attempts < max_attempts:
                start = (attempts * self.TRIPLETS_PER_CALL) % max(1, len(matching))
                batch = matching[start: start + self.TRIPLETS_PER_CALL]
                if len(batch) < 3:
                    batch = matching[:self.TRIPLETS_PER_CALL]

                attempts += 1
                print(f"      [进度 {topic_count+1}/{needed}] 生成中 ...", end="", flush=True)
                raw_q = self._generate_one(topic, batch)

                if not raw_q:
                    print(" ✗ 生成为空（JSON 解析失败）")
                    continue

                q_preview = raw_q.get("question", "")[:60].replace("\n", " ")
                a_preview = raw_q.get("correct_answer", "")[:30]
                print(f" Q: {q_preview}... | A: {a_preview}", end="", flush=True)

                if self._verify_question(raw_q):
                    bank_q = self._to_bank_format(raw_q, tname, len(self.questions))
                    self.questions.append(bank_q)
                    topic_count += 1
                    total_new  += 1
                    print(" ✓")
                else:
                    print(" ✗ 验证不通过（答案有误）")

                self._save()
                time.sleep(1.0)

            print(f"      '{tname}' 累计生成 {topic_count} 道")

        self._save()
        kg_total = sum(1 for q in self.questions if q.get('source') == 'kg_generated')
        print(f"\n{'='*50}")
        print(f"KG 题库构建完成！本次新增 {total_new} 道，KG 生成总计 {kg_total} 道")
        print(f"已保存至: {self.output_file}")
        print('='*50)


# ==========================================
# 3b. 多题型题库构建器 (NEW)
# ==========================================

FORMATS = ["mcq_single", "mcq_multi", "true_false", "fill_blank", "open_answer"]
N_PER_FORMAT = 2  # 每个 topic × format 生成的题目数量

MT_TARGET_TOPICS = [
    # ---- 算法类（计算题为主）----
    {
        "name": "hash table linear probing",
        "primary_type": "computational",
        "keywords": [
            "hash", "linear prob", "h(k)", "collision", "open addressing",
            "load factor", "cluster", "primary cluster", "worst case",
            "deletion", "tombstone", "rehash", "expected probe",
        ],
        "conceptual_angle": "hash tables and collision resolution trade-offs in system design",
    },
    {
        "name": "hash table quadratic probing",
        "primary_type": "computational",
        "keywords": [
            "quadratic prob", "hash", "double hash", "secondary clustering",
            "load factor", "collision", "probe sequence", "worst case",
        ],
        "conceptual_angle": "comparison of linear vs quadratic probing strategies",
    },
    {
        "name": "recursion and recurrence",
        "primary_type": "computational",
        "keywords": ["recursion", "recurrence", "base case", "T(n)", "recursive", "call stack"],
        "conceptual_angle": "when to choose recursion vs iteration, stack overflow scenarios",
    },
    {
        "name": "merge sort",
        "primary_type": "computational",
        "keywords": ["merge sort", "mergesort", "divide and conquer", "merge step"],
        "conceptual_angle": "merge sort stability and external sorting use cases",
    },
    {
        "name": "binary search tree operations",
        "primary_type": "computational",
        "keywords": ["binary search tree", "bst", "inorder", "predecessor", "successor"],
        "conceptual_angle": "BST performance degradation and when to use balanced trees",
    },
    {
        "name": "BFS graph traversal",
        "primary_type": "computational",
        "keywords": ["bfs", "breadth-first", "queue", "visited", "shortest path unweighted"],
        "conceptual_angle": "BFS vs DFS choice for different problem types",
    },
    {
        "name": "dynamic programming knapsack",
        "primary_type": "computational",
        "keywords": ["dynamic programming", "dp table", "knapsack", "subproblem", "memoization"],
        "conceptual_angle": "identifying overlapping subproblems and optimal substructure",
    },
    # ---- 概念类（概念题为主）----
    {
        "name": "cybersecurity",
        "primary_type": "conceptual",
        "keywords": ["encryption", "authentication", "vulnerability", "attack", "CIA triad", "firewall"],
        "conceptual_angle": "real-world security decisions: protocol choice, threat modeling",
    },
    {
        "name": "computer networks",
        "primary_type": "conceptual",
        "keywords": ["TCP", "UDP", "routing", "DNS", "HTTP", "latency", "bandwidth", "OSI"],
        "conceptual_angle": "protocol selection and network troubleshooting scenarios",
    },
    {
        "name": "operating systems",
        "primary_type": "conceptual",
        "keywords": ["process", "thread", "deadlock", "scheduling", "virtual memory", "paging"],
        "conceptual_angle": "OS resource management trade-offs and failure scenarios",
    },
    {
        "name": "machine learning",
        "primary_type": "conceptual",
        "keywords": ["overfitting", "bias variance", "gradient descent", "loss function", "regularization"],
        "conceptual_angle": "diagnosing model problems and choosing appropriate algorithms",
    },
    {
        "name": "database systems",
        "primary_type": "conceptual",
        "keywords": ["ACID", "normalization", "index", "transaction", "SQL", "join", "foreign key"],
        "conceptual_angle": "database design decisions and query performance scenarios",
    },
]


def _mt_fmt_instruction(fmt: str, topic_name: str, primary_type: str, conceptual_angle: str) -> str:
    """Return LLM prompt for a given question format."""
    conceptual_guide = ""
    if primary_type == "conceptual" or fmt in ("open_answer", "mcq_multi"):
        conceptual_guide = f"""
=== QUALITY STANDARD FOR CONCEPTUAL QUESTIONS ===
BAD (never generate these):
  ✗ "What is {topic_name}?"  — pure definition
  ✗ "Define the term X."

GOOD (use these styles):
  ✓ SCENARIO JUDGMENT: Give a concrete situation, ask which approach is better/what will happen
  ✓ CONCEPT DISCRIMINATION: Compare two easily confused concepts
  ✓ FAULT DIAGNOSIS: Describe a symptom, ask for root cause
  ✓ CROSS-CONCEPT REASONING: Require connecting ≥2 concepts to answer

Angle: {conceptual_angle}
"""

    if fmt == "mcq_single":
        if primary_type == "computational":
            return f"""Generate ONE computational single-choice question about "{topic_name}".

FORBIDDEN: Do NOT generate a basic "insert N keys and find where key X lands" question.
Such a question tests only one step and will score 1/5 on depth.

REQUIRED — your question MUST chain at least two of the knowledge graph relations above.
Example of a good chain for hash tables:
  Step 1: compute load factor after insertions  (uses HAS_COMPLEXITY relation)
  Step 2: use load factor to determine expected probe count for next insertion  (uses PRODUCES_OUTPUT relation)
  -> Question: "After inserting these keys, what is the load factor AND the worst-case probes for the next insertion?"

RULES:
- Tiny concrete inputs (table size <= 8, at most 5 keys, numbers <= 30)
- Compute answer step-by-step in "scratchpad" FIRST
- Each distractor maps to a SPECIFIC error: [Boundary Error] / [Initialization Error] / [Procedural Omission] / [Operator Confusion]
- "answer": plain value only, NO option letter (e.g. "4 probes" not "B. 4 probes")
- "options": list of 4 strings like ["A. ...", "B. ...", "C. ...", "D. ..."]
- The correct answer MUST appear as one of the options

Return ONLY this JSON:
{{"scratchpad":"Step 1:...","question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"...","distractors":[{{"option":"A. ...","error_tag":"[Initialization Error]","explanation":"Student..."}}],"rationale":"..."}}"""
        else:
            return f"""{conceptual_guide}
Generate ONE conceptual single-choice question about "{topic_name}".
RULES:
- Follow GOOD question styles above (scenario/discrimination/diagnosis)
- "answer": letter only e.g. "A"
- "options": list of 4 strings ["A. ...", "B. ...", "C. ...", "D. ..."]
- Wrong options represent common real misconceptions

Return ONLY this JSON:
{{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","distractors":[{{"option":"B. ...","error_tag":"[Misconception]","explanation":"..."}}],"rationale":"..."}}"""

    elif fmt == "mcq_multi":
        if primary_type == "computational":
            return f"""Generate ONE multiple-select computational question about "{topic_name}" where 2 or 3 answers are correct.

FORBIDDEN: Do NOT ask "which keys collide" or "which index does key X land at".
These test only one step.

REQUIRED: The question must combine at least two computational concepts from the
knowledge graph relations above.
Good examples for hash tables:
  - "Which of the following statements about a hash table with load factor 0.8 are true?"
    (combines load factor computation + clustering effect + probe count)
  - "After inserting [10,20,30] with h(k)=k mod 5, which statements correctly
    describe the table state AND its expected search performance?"

RULES:
- Stem must say "(Select ALL that apply)"
- Exactly 4 options A-D; 2-3 must be correct
- Wrong options represent plausible but incorrect beliefs about algorithm behaviour

Return ONLY this JSON:
{{"question":"... (Select ALL that apply)","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answers":["A","C"],"explanation":"A is correct because... B is wrong because..."}}"""
        else:
            return f"""{conceptual_guide}
Generate ONE multiple-select question about "{topic_name}" where 2 or 3 answers are correct.
RULES:
- Stem must say "(Select ALL that apply)"
- Exactly 4 options A-D; 2-3 must be correct
- Wrong options: plausible but incorrect beliefs

Return ONLY this JSON:
{{"question":"... (Select ALL that apply)","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answers":["A","C"],"explanation":"A is correct because... B is wrong because..."}}"""

    elif fmt == "true_false":
        if primary_type == "computational":
            return f"""Generate ONE True/False question about "{topic_name}" based on a non-trivial algorithmic consequence.

FORBIDDEN: Do NOT make a statement that only requires tracing one insertion step.
  Bad: "After inserting key 14 with h(k)=k mod 7, it lands at index 0. True/False"

REQUIRED: The statement must assert a consequence that depends on at least two
computational steps — ideally a claim about performance, load factor,
clustering behaviour, or worst-case complexity derived from a concrete scenario.
  Good: "A hash table of size 5 with h(k)=k mod 5 after inserting [5,10,15]
has load factor 0.6 and requires at most 3 probes to insert the next key
that hashes to slot 0. True/False"

RULES:
- Statement must be fully self-contained (give the concrete input)
- If FALSE, the explanation must state the CORRECT value or behaviour

Return ONLY this JSON:
{{"statement":"...","tf_answer":true,"explanation":"This is TRUE/FALSE because... The correct behaviour is..."}}"""
        else:
            return f"""{conceptual_guide}
Generate ONE True/False question about "{topic_name}" that tests deep understanding.
RULES:
- Statement involves a consequence, trade-off, or scenario (NOT a definition)
- If FALSE, clearly explain what the correct relationship is

Return ONLY this JSON:
{{"statement":"...","tf_answer":false,"explanation":"This is TRUE/FALSE because..."}}"""

    elif fmt == "fill_blank":
        if primary_type == "computational":
            return f"""Generate ONE fill-in-the-blank question about "{topic_name}" requiring multi-step computation.

FORBIDDEN: Do NOT use a single blank that only asks for a final slot index.
  Bad: "After inserting 14 with h(k)=k mod 7, it is stored at slot [BLANK_1]."

REQUIRED: Use exactly 2 blanks that correspond to TWO DIFFERENT computational steps
chained together — the second blank's value should logically depend on the first.
  Good: "After inserting keys [5, 10, 15] into a hash table of size 5 with
h(k)=k mod 5, the load factor is [BLANK_1] and the worst-case number of probes
to insert the next key hashing to slot 0 is [BLANK_2]."
  (BLANK_1 = 0.6, BLANK_2 = 4 — BLANK_2 depends on knowing BLANK_1 first)

RULES:
- Use [BLANK_1], [BLANK_2] in "sentence" (exactly 2 blanks)
- "answers" list order matches blank numbering
- Include "scratchpad" showing computation for both blanks

Return ONLY this JSON:
{{"sentence":"... [BLANK_1] ... [BLANK_2] ...","answers":["value1","value2"],"scratchpad":"Step 1: compute load factor... Step 2: use load factor to derive...","explanation":"..."}}"""
        else:
            return f"""{conceptual_guide}
Generate ONE fill-in-the-blank sentence about "{topic_name}" testing key terminology in context.
RULES:
- Use [BLANK_1], [BLANK_2] (1-2 blanks)
- Each blank is a technical term uniquely determined by context
- Sentence describes a mechanism or consequence, not just a definition

Return ONLY this JSON:
{{"sentence":"... [BLANK_1] ... [BLANK_2] ...","answers":["term1","term2"],"explanation":"..."}}"""

    elif fmt == "open_answer":
        if primary_type == "computational":
            return f"""Generate ONE open-answer question about "{topic_name}" that requires both computation and conceptual reasoning.

FORBIDDEN: Do NOT ask "What is X?" or "Describe how linear probing works."

REQUIRED: Give a concrete scenario (specific table size, hash function, input keys)
and ask the student to both compute a result AND explain the algorithmic consequence.
  Good: "A hash table of size 7 uses h(k)=k mod 7 with linear probing.
After inserting keys [0, 7, 14, 21], compute the load factor and explain
how primary clustering affects the expected number of probes for the next
insertion. What would change if quadratic probing were used instead?"

RULES:
- "model_answer": must include both a computed value and a conceptual explanation
- "key_points": 3-4 criteria; at least one must be a specific computed value,
  at least one must describe a causal relationship between two concepts

Return ONLY this JSON:
{{"question":"...","model_answer":"...","key_points":["Correctly computes load factor as X/N = ...","Explains how primary clustering increases expected probe count beyond O(1)","Correctly identifies that quadratic probing reduces primary clustering but risks cycling"]}}"""
        else:
            return f"""{conceptual_guide}
Generate ONE open-answer question about "{topic_name}".
RULES:
- Follow GOOD question styles: scenario/diagnosis/cross-concept reasoning
- Question requires >=2 key concepts to answer well
- "model_answer": 3-5 substantive sentences
- "key_points": 3-4 specific technical criteria a good answer must hit
- DO NOT ask "What is X?" or "Define X."

Return ONLY this JSON:
{{"question":"...","model_answer":"...","key_points":["Mentions X and explains why...","Correctly describes trade-off between Y and Z","Identifies failure condition and specific mitigation"]}}"""

    return ""


class MultiTypeQuestionBankBuilder:
    """
    Generates 5 question formats × N_PER_FORMAT questions × all target topics.
    Appends results to the existing question_bank.json.
    """

    def __init__(self, api_key: str, kg_path: str, output_file: str):
        self.llm         = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.kg_path     = kg_path
        self.output_file = output_file
        self.questions: List[Dict] = []
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    self.questions = json.load(f)
                mt = sum(1 for q in self.questions if q.get("source") == "kg_multitype")
                print(f"[*] 已加载题库 {len(self.questions)} 道（multitype: {mt} 道）")
            except Exception as e:
                print(f"[!] 读取题库失败: {e}")

    def _save(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.questions, f, indent=2, ensure_ascii=False)

    def _load_triplets(self) -> List[Dict]:
        if not os.path.exists(self.kg_path):
            return []
        with open(self.kg_path, "r", encoding="utf-8") as f:
            return json.load(f).get("triplets", [])

    def _filter_triplets(self, triplets: List[Dict], keywords: List[str]) -> List[Dict]:
        result = []
        for t in triplets:
            combined = (t.get("head", "") + " " + t.get("tail", "")).lower()
            if any(kw.lower() in combined for kw in keywords):
                result.append(t)
        return result

    def _already_covered(self, topic_name: str, fmt: str) -> bool:
        existing = [
            q for q in self.questions
            if q.get("source") == "kg_multitype"
            and q.get("topic", "").lower() == topic_name.lower()
            and q.get("question_format") == fmt
        ]
        return len(existing) >= N_PER_FORMAT

    def _generate_one(self, topic: Dict, fmt: str, triplet_batch: List[Dict], q_type: str):
        if triplet_batch:
            triplet_str = "=== RELEVANT KNOWLEDGE FROM TEXTBOOK ===\n" + "\n".join(
                f"  ({t['head'][:60]}) --[{t['relation']}]--> ({t['tail'][:80]})"
                for t in triplet_batch[:8]
            ) + "\n\n"
        else:
            triplet_str = ""

        prompt = triplet_str + _mt_fmt_instruction(
            fmt, topic["name"], q_type, topic["conceptual_angle"]
        )

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1600,
                temperature=0.85,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"    [!] 生成失败 ({fmt}): {e}")
            return None

    def _verify_computational(self, q_dict: Dict, fmt: str) -> bool:
        if fmt == "mcq_single":
            question_text = q_dict.get("question", "")
            stated_answer = q_dict.get("answer", "")
        elif fmt == "fill_blank":
            question_text = q_dict.get("sentence", "").replace("[BLANK_1]", "___").replace("[BLANK_2]", "___").replace("[BLANK_3]", "___")
            stated_answer = str(q_dict.get("answers", [""])[0])
        else:
            return True

        if not question_text or not stated_answer:
            return False

        verify_prompt = (
            f"Solve this CS problem independently, showing all work step by step.\n\n"
            f"{question_text}\n\n"
            f"At the very end write ONE line:\n"
            f"MY_ANSWER: <your computed result>"
        )
        try:
            res = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0, max_tokens=800,
            )
            text = res.choices[0].message.content
            match = re.search(r"MY_ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            if not match:
                return True
            computed = re.sub(r"^[a-d][.\)]\s*", "", match.group(1).strip().lower()).strip()
            expected = re.sub(r"^[a-d][.\)]\s*", "", stated_answer.strip().lower()).strip()
            if computed == expected or expected in computed or computed in expected:
                return True
            try:
                if abs(float(computed.split()[0]) - float(expected.split()[0])) < 0.01:
                    return True
            except (ValueError, TypeError, IndexError):
                pass
            return False
        except Exception:
            return True

    def _is_valid(self, raw_q: Dict, fmt: str) -> bool:
        if fmt == "mcq_single":
            return bool(raw_q.get("question")) and bool(raw_q.get("options")) and bool(raw_q.get("answer"))
        elif fmt == "mcq_multi":
            return bool(raw_q.get("question")) and bool(raw_q.get("options")) and bool(raw_q.get("correct_answers"))
        elif fmt == "true_false":
            return bool(raw_q.get("statement")) and raw_q.get("tf_answer") is not None
        elif fmt == "fill_blank":
            return bool(raw_q.get("sentence")) and bool(raw_q.get("answers"))
        elif fmt == "open_answer":
            return bool(raw_q.get("question")) and bool(raw_q.get("model_answer"))
        return False

    def _to_bank_format(self, raw_q: Dict, topic: Dict, fmt: str, idx: int, q_type: str) -> Dict:
        slug = topic["name"].replace(" ", "_").replace("/", "_")
        return {
            "id":              f"mt_{slug}_{fmt}_{q_type[:4]}_{idx:03d}",
            "question_format": fmt,
            "question_type":   q_type,
            "type":            q_type,
            "topic":           topic["name"],
            "source":          "kg_multitype",
            "question":        raw_q.get("question", raw_q.get("statement", raw_q.get("sentence", ""))),
            "answer":          raw_q.get("answer", ""),
            "options":         raw_q.get("options", []),
            "rationale":       raw_q.get("rationale", raw_q.get("explanation", "")),
            "scratchpad":      raw_q.get("scratchpad", ""),
            "statement":       raw_q.get("statement", ""),
            "sentence":        raw_q.get("sentence", ""),
            "correct_answers": raw_q.get("correct_answers", []),
            "answers":         raw_q.get("answers", []),
            "tf_answer":       raw_q.get("tf_answer", None),
            "distractors":     raw_q.get("distractors", []),
            "model_answer":    raw_q.get("model_answer", ""),
            "key_points":      raw_q.get("key_points", []),
            "explanation":     raw_q.get("explanation", ""),
        }

    def build(self):
        print(f"\n{'='*60}")
        print("MultiTypeQuestionBankBuilder — 5 种题型生成")
        print(f"目标: {len(MT_TARGET_TOPICS)} topics × {len(FORMATS)} formats × {N_PER_FORMAT} 道")
        print(f"{'='*60}")

        all_triplets = self._load_triplets()
        print(f"[*] 已加载 {len(all_triplets)} 条 KG triplet")

        total_new = 0

        for topic in MT_TARGET_TOPICS:
            tname = topic["name"]
            matching = self._filter_triplets(all_triplets, topic["keywords"])
            random.shuffle(matching)

            for q_type in ["computational", "conceptual"]:
                print(f"\n>>> {tname}  ({q_type})  — {len(matching)} 条 triplet")

                for fmt in FORMATS:
                    # 动态计算该 topic × format × q_type 组合需要补充的数量
                    existing_count = sum(
                        1 for q in self.questions
                        if q.get("source") == "kg_multitype"
                        and q.get("topic", "").lower() == tname.lower()
                        and q.get("question_format") == fmt
                        and (q.get("type") == q_type or q.get("question_type") == q_type)
                    )
                    needed = N_PER_FORMAT - existing_count

                    if needed <= 0:
                        print(f"   [{fmt:12s}] 已足够，跳过")
                        continue

                    print(f"   [{fmt:12s}]", end=" ", flush=True)
                    generated = 0
                    attempts  = 0
                    max_attempts = needed * 3

                    while generated < needed and attempts < max_attempts:
                        attempts += 1
                        start = ((attempts - 1) * 8) % max(1, len(matching))
                        batch = matching[start: start + 8] if matching else []

                        raw_q = self._generate_one(topic, fmt, batch, q_type)
                        if raw_q is None:
                            print("✗gen", end=" ", flush=True); time.sleep(1); continue

                        if not self._is_valid(raw_q, fmt):
                            print("✗schema", end=" ", flush=True); time.sleep(0.5); continue

                        if q_type == "computational" and fmt in ("mcq_single", "fill_blank"):
                            if not self._verify_computational(raw_q, fmt):
                                print("✗verify", end=" ", flush=True); time.sleep(1); continue

                        entry = self._to_bank_format(raw_q, topic, fmt, len(self.questions), q_type)
                        self.questions.append(entry)
                        self._save()
                        generated  += 1
                        total_new  += 1
                        print("✓", end=" ", flush=True)
                        time.sleep(1.2)

                    print()

        self._save()
        mt_total = sum(1 for q in self.questions if q.get("source") == "kg_multitype")
        print(f"\n{'='*60}")
        print(f"完成！本次新增 {total_new} 道")
        print(f"multitype 累计: {mt_total} 道 | 题库总计: {len(self.questions)} 道")
        print(f"{'='*60}")


# ==========================================
# 4. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['all', 'kg_only', 'qb_only', 'extract_only', 'multi_type'],
        default='all',
        help=(
            'all         — run everything (default)\n'
            'extract_only — only extract KG from textbooks\n'
            'qb_only     — only rebuild question bank (no KG extraction)\n'
            'kg_only     — only run KGBasedQuestionBuilder\n'
            'multi_type  — only run MultiTypeQuestionBankBuilder (5 question formats)\n'
        )
    )
    args = parser.parse_args()

    textbook_path  = "./GraphRAG-Bench/textbooks"
    questions_path = "./GraphRAG-Bench/questions"
    output_kg_path = "./global_knowledge_graph.json"
    output_qb_path = "./question_bank.json"

    # Step 1: KG extraction from textbooks
    if args.mode in ('all', 'extract_only'):
        extractor = LLMGraphExtractor(API_KEY, output_kg_path)
        if os.path.exists(textbook_path):
            extractor.process_dataset(textbook_path)
        else:
            print(f"找不到教材文件夹: {textbook_path}，跳过知识图谱构建。")

    if args.mode == 'extract_only':
        return

    # Step 2: Classify existing dataset questions
    if args.mode in ('all', 'qb_only'):
        qb_builder = QuestionBankBuilder(API_KEY, output_qb_path)
        if os.path.exists(questions_path):
            qb_builder.build(questions_path)
        else:
            print(f"找不到题目文件夹: {questions_path}，跳过题库构建。")

    # Step 3: Generate computational questions from KG
    if args.mode in ('all', 'qb_only', 'kg_only'):
        if os.path.exists(output_kg_path):
            kg_qb = KGBasedQuestionBuilder(API_KEY, output_kg_path, output_qb_path)
            kg_qb.build()
        else:
            print(f"找不到 KG 文件: {output_kg_path}，跳过 KG 题目生成。")

    # Step 4: Generate multi-type questions (NEW)
    if args.mode in ('all', 'qb_only', 'multi_type'):
        if os.path.exists(output_kg_path):
            mt_builder = MultiTypeQuestionBankBuilder(API_KEY, output_kg_path, output_qb_path)
            mt_builder.build()
        else:
            print(f"找不到 KG 文件: {output_kg_path}，跳过多题型生成。")

    # Final summary
    if os.path.exists(output_qb_path):
        with open(output_qb_path) as f:
            qs = json.load(f)
        from collections import Counter as _Counter
        comp   = sum(1 for q in qs if q.get('type') == 'computational' or q.get('question_type') == 'computational')
        kg_gen = sum(1 for q in qs if q.get('source') == 'kg_generated')
        mt_gen = sum(1 for q in qs if q.get('source') == 'kg_multitype')
        fmt_counts = dict(_Counter(q.get('question_format', 'legacy') for q in qs))
        print(f"\n[SUMMARY] 题库总计: {len(qs)} 道")
        print(f"  计算题: {comp} 道")
        print(f"  KG 单题型生成: {kg_gen} 道 | KG 多题型生成: {mt_gen} 道")
        print(f"  题型分布: {fmt_counts}")


if __name__ == "__main__":
    main()