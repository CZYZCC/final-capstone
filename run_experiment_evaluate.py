"""
run_experiment_evaluate.py
Independent Evaluation Script for the generated questions.
Usage:
    python run_experiment_evaluate.py --input experiment_data/raw_graph_rag.jsonl --output experiment_data/scored_graph_rag.jsonl
"""

import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the evaluator
from rag_system.evaluator import AutomatedEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generated Questions")
    parser.add_argument("--input", type=str, required=True, help="Path to raw JSONL file (e.g., raw_graph_rag.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to save scored JSONL file")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

    print("[*] Initializing Automated Evaluator...")
    llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    evaluator = AutomatedEvaluator(llm_client)

    if not os.path.exists(args.input):
        print(f"[!] Input file not found: {args.input}")
        return

    # ==========================================
    # CHECKPOINTING LOGIC
    # ==========================================
    scored_questions = set()
    if os.path.exists(args.output):
        print(f"[*] Found existing evaluation output '{args.output}'. Loading checkpoints...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Use a hash of the raw question dictionary as a unique identifier
                    q_hash = hash(json.dumps(data.get("question", {}), sort_keys=True))
                    scored_questions.add(q_hash)
                except json.JSONDecodeError:
                    continue
        print(f"[*] Skipping {len(scored_questions)} previously evaluated questions.")

    # ==========================================
    # LOAD PENDING TASKS
    # ==========================================
    pending_tasks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                q_hash = hash(json.dumps(item, sort_keys=True))
                if q_hash not in scored_questions:
                    pending_tasks.append(item)
            except json.JSONDecodeError:
                continue

    if not pending_tasks:
        print("[*] All questions in the input file have already been evaluated!")
        return

    print(f"[*] {len(pending_tasks)} questions remaining to be evaluated.")

    # ==========================================
    # EVALUATION LOOP
    # ==========================================
    with open(args.output, 'a', encoding='utf-8') as out_f:
        pbar = tqdm(pending_tasks, desc="Evaluating", unit="q")
        
        for item in pbar:
            topic = item.get("topic", "Unknown")
            q_format = item.get("question_format", "Unknown")
            
            # The evaluator expects the stringified item
            q_json_str = json.dumps(item, ensure_ascii=False)
            
            try:
                # We pass an empty list for context since the generated JSON already holds everything
                score_result = evaluator.evaluate(question_json=q_json_str, context=[], topic=topic)
                
                # Assemble the final payload
                final_item = {
                    "metadata": {
                        "topic": topic,
                        "model": item.get("model", "unknown"),
                        "method": item.get("method", "unknown"),
                        "question_format": q_format,
                        "expected_type": item.get("expected_type", "unknown"),
                        "generation_time": item.get("generation_time", 0)
                    },
                    "question": item,
                    "score": score_result
                }
                
                out_f.write(json.dumps(final_item, ensure_ascii=False) + "\n")
                out_f.flush()
                
            except Exception as e:
                print(f"\n[!] Evaluation failed for topic '{topic}': {str(e)}")
                # Continue evaluating the rest even if one fails
                continue

    print(f"\n[*] Evaluation complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()