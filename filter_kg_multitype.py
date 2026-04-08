"""
filter_kg_multitype.py
======================
对 question_bank.json 中 source=kg_multitype 的题目进行 LLM 评分筛选。

输出三个文件：
  - kept.json        : 评分通过，直接保留
  - needs_review.json: 评分边界，需人工复查
  - rejected.json    : 评分不通过，直接丢弃
  - scores_log.json  : 所有题目的原始评分记录（可断点续评）

运行：
  python filter_kg_multitype.py --qb question_bank.json --api_key YOUR_KEY
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

from openai import OpenAI

# ── 阈值 ──────────────────────────────────────────────────────────────
AUTO_KEEP_MIN   = 4.0   # 三项均值 >= 4.0 → 自动保留
AUTO_REJECT_MAX = 2.5   # 三项均值 <= 2.5 → 自动丢弃
# 其余 → needs_review
# ──────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# 1. 格式有效性检查（不调 LLM，纯规则）
# ══════════════════════════════════════════════════════════════════════

def _is_format_valid(q: Dict) -> Tuple[bool, str]:
    """
    返回 (is_valid, reason)。
    只检查「关键字段是否存在且非空」，不判断内容对错。
    """
    fmt = q.get("question_format", "")

    if fmt == "mcq_single":
        if not q.get("question"):
            return False, "missing question"
        opts = q.get("options", [])
        if len(opts) < 4:
            return False, f"options count={len(opts)} < 4"
        ans = q.get("answer", "")
        if not ans:
            return False, "missing answer"

    elif fmt == "mcq_multi":
        if not q.get("question"):
            return False, "missing question"
        opts = q.get("options", {})
        if len(opts) < 4:
            return False, f"options count={len(opts)} < 4"
        ca = q.get("correct_answers", [])
        if not ca:
            return False, "missing correct_answers"

    elif fmt == "true_false":
        stmt = q.get("statement") or q.get("question", "")
        if not stmt:
            return False, "missing statement"
        tf = q.get("tf_answer")
        if tf is None:
            return False, "missing tf_answer"

    elif fmt == "fill_blank":
        sent = q.get("sentence") or q.get("question", "")
        if not sent:
            return False, "missing sentence"
        answers = q.get("answers", [])
        if not answers:
            return False, "missing answers"

    elif fmt == "open_answer":
        if not q.get("question"):
            return False, "missing question"
        if not q.get("model_answer"):
            return False, "missing model_answer"

    else:
        return False, f"unknown format: {fmt}"

    return True, "ok"


# ══════════════════════════════════════════════════════════════════════
# 2. 把题目序列化成 LLM 可读的文本
# ══════════════════════════════════════════════════════════════════════

def _question_to_text(q: Dict) -> str:
    fmt   = q.get("question_format", "")
    topic = q.get("topic", "")
    qtype = q.get("type") or q.get("question_type", "")

    lines = [
        f"Topic   : {topic}",
        f"Type    : {qtype}",
        f"Format  : {fmt}",
        "",
    ]

    if fmt == "mcq_single":
        lines.append(f"Question: {q.get('question','')}")
        for opt in q.get("options", []):
            lines.append(f"  {opt}")
        lines.append(f"Answer  : {q.get('answer','')}")
        if q.get("rationale"):
            lines.append(f"Rationale: {q.get('rationale','')[:300]}")

    elif fmt == "mcq_multi":
        lines.append(f"Question: {q.get('question','')}")
        opts = q.get("options", {})
        if isinstance(opts, dict):
            for k, v in opts.items():
                lines.append(f"  {k}. {v}")
        else:
            for opt in opts:
                lines.append(f"  {opt}")
        lines.append(f"Correct answers: {q.get('correct_answers','')}")
        if q.get("explanation"):
            lines.append(f"Explanation: {q.get('explanation','')[:300]}")

    elif fmt == "true_false":
        stmt = q.get("statement") or q.get("question", "")
        lines.append(f"Statement: {stmt}")
        lines.append(f"Answer   : {q.get('tf_answer','')}")
        if q.get("explanation"):
            lines.append(f"Explanation: {q.get('explanation','')[:300]}")

    elif fmt == "fill_blank":
        sent = q.get("sentence") or q.get("question", "")
        lines.append(f"Sentence : {sent}")
        lines.append(f"Answers  : {q.get('answers','')}")
        if q.get("scratchpad"):
            lines.append(f"Scratchpad: {q.get('scratchpad','')[:300]}")

    elif fmt == "open_answer":
        lines.append(f"Question    : {q.get('question','')}")
        lines.append(f"Model answer: {q.get('model_answer','')[:400]}")
        if q.get("key_points"):
            lines.append(f"Key points  : {q.get('key_points','')}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 3. LLM 评分
# ══════════════════════════════════════════════════════════════════════

SCORE_PROMPT = """You are an expert CS exam designer and quality reviewer.
Evaluate the following CS exam question on THREE dimensions, each scored 1–5:

1. TYPE_FIT (1-5)
   Does the question genuinely match its stated type?
   - "computational": must require actual calculation, algorithm tracing, or
     step-by-step numeric/symbolic derivation. Score 1 if it only asks for
     definitions or concepts with no computation required.
   - "conceptual": must test understanding through scenario judgment,
     concept discrimination, or fault diagnosis. Score 1 if it is a pure
     definition question ("What is X?") or if it actually requires computation.
   Note: open_answer format questions follow the same computational/conceptual
   distinction — score based on whether computation or conceptual reasoning
   is primarily required.

2. ANSWER_CORRECT (1-5)
   Is the stated answer/correct_answer factually correct?
   Score 5 = definitely correct, 3 = plausible but uncertain, 1 = clearly wrong.
   For open_answer, judge whether the model_answer is accurate and complete.

3. QUALITY (1-5)
   Is this a good exam question?
   Score 5 = discriminative, clear, meaningful distractors/wrong options.
   Score 3 = acceptable but could be improved.
   Score 1 = trivial ("What is X?"), ambiguous, or distractors are obviously wrong.

Return ONLY this JSON (no extra text):
{{
  "type_fit": <1-5>,
  "answer_correct": <1-5>,
  "quality": <1-5>,
  "type_fit_reason": "<one sentence>",
  "answer_correct_reason": "<one sentence>",
  "quality_reason": "<one sentence>"
}}

--- QUESTION TO EVALUATE ---
{question_text}
"""


def _score_question(llm: OpenAI, q: Dict) -> Dict:
    q_text = _question_to_text(q)
    prompt = SCORE_PROMPT.format(question_text=q_text)

    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=400,
        )
        data = json.loads(resp.choices[0].message.content)
        data["avg"] = round(
            (data.get("type_fit", 3) +
             data.get("answer_correct", 3) +
             data.get("quality", 3)) / 3, 2
        )
        return data
    except Exception as e:
        return {"error": str(e), "avg": 3.0}


# ══════════════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qb",      default="./question_bank.json")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--out_dir", default="./filter_output")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scores_log_path = os.path.join(args.out_dir, "scores_log.json")
    kept_path       = os.path.join(args.out_dir, "kept.json")
    review_path     = os.path.join(args.out_dir, "needs_review.json")
    rejected_path   = os.path.join(args.out_dir, "rejected.json")

    # ── 加载题库 ──────────────────────────────────────────────────────
    with open(args.qb, encoding="utf-8") as f:
        all_questions: List[Dict] = json.load(f)

    mt_questions = [q for q in all_questions if q.get("source") == "kg_multitype"]
    print(f"[*] kg_multitype 题目共 {len(mt_questions)} 道")

    # ── 加载已有评分记录（断点续评）───────────────────────────────────
    scores_log: Dict[str, Dict] = {}
    if os.path.exists(scores_log_path):
        with open(scores_log_path, encoding="utf-8") as f:
            scores_log = json.load(f)
        print(f"[*] 已有评分记录 {len(scores_log)} 条，跳过已评分题目")

    llm = OpenAI(api_key=args.api_key, base_url=args.base_url)

    kept, review, rejected = [], [], []

    # ── 统计格式无效（不耗 LLM）─────────────────────────────────────
    format_invalid = []

    for i, q in enumerate(mt_questions):
        qid = q.get("id", f"idx_{i}")

        # Step 1: 格式有效性
        valid, reason = _is_format_valid(q)
        if not valid:
            print(f"  [{i+1}/{len(mt_questions)}] {qid} → ✗ 格式无效: {reason}")
            entry = {**q, "_filter": {"verdict": "rejected", "reason": f"format_invalid: {reason}"}}
            rejected.append(entry)
            format_invalid.append(qid)
            continue

        # Step 2: LLM 评分（已评过则跳过）
        if qid not in scores_log:
            print(f"  [{i+1}/{len(mt_questions)}] {qid} 评分中...", end=" ", flush=True)
            score = _score_question(llm, q)
            scores_log[qid] = score
            # 实时保存评分日志
            with open(scores_log_path, "w", encoding="utf-8") as f:
                json.dump(scores_log, f, indent=2, ensure_ascii=False)
            time.sleep(0.8)
        else:
            score = scores_log[qid]
            print(f"  [{i+1}/{len(mt_questions)}] {qid} (已缓存)", end=" ", flush=True)

        avg = score.get("avg", 3.0)
        tf  = score.get("type_fit", 3)
        ac  = score.get("answer_correct", 3)
        ql  = score.get("quality", 3)

        # Step 3: 判决
        entry = {
            **q,
            "_scores": {
                "type_fit":        tf,
                "answer_correct":  ac,
                "quality":         ql,
                "avg":             avg,
                "type_fit_reason":       score.get("type_fit_reason", ""),
                "answer_correct_reason": score.get("answer_correct_reason", ""),
                "quality_reason":        score.get("quality_reason", ""),
            }
        }

        if avg >= AUTO_KEEP_MIN and tf >= 3 and ac >= 3:
            verdict = "kept"
            kept.append(entry)
        elif avg <= AUTO_REJECT_MAX or ac <= 2 or tf <= 2:
            verdict = "rejected"
            rejected.append(entry)
        else:
            verdict = "needs_review"
            review.append(entry)

        entry["_scores"]["verdict"] = verdict
        print(f"avg={avg:.1f} (tf={tf} ac={ac} ql={ql}) → {verdict}")

    # ── 保存结果 ─────────────────────────────────────────────────────
    def _save(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    _save(kept_path,    kept)
    _save(review_path,  review)
    _save(rejected_path, rejected)
    _save(scores_log_path, scores_log)

    # ── 汇总 ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"筛选完成！共 {len(mt_questions)} 道 kg_multitype 题")
    print(f"  ✓ 自动保留   : {len(kept):3d} 道  → {kept_path}")
    print(f"  ? 需人工复查 : {len(review):3d} 道  → {review_path}")
    print(f"  ✗ 自动丢弃   : {len(rejected):3d} 道  → {rejected_path}")
    print(f"    其中格式无效: {len(format_invalid):3d} 道")
    print(f"{'='*55}")

    # ── 按 topic+type 汇总保留情况 ───────────────────────────────────
    from collections import defaultdict, Counter
    topic_type_kept = defaultdict(Counter)
    for q in kept + review:   # review 也算潜在保留
        topic_type_kept[q.get("topic","?")][
            q.get("type") or q.get("question_type","?")
        ] += 1

    print("\n保留+待复查 题目的 topic×type 分布：")
    for topic in sorted(topic_type_kept):
        print(f"  {topic}: {dict(topic_type_kept[topic])}")


if __name__ == "__main__":
    main()
