"""
run_experiment_evaluate.py
Independent Evaluation Script with Checkpointing, Isolated Logging, and Memory Restoration.
Usage:
    python run_experiment_evaluate.py --model graph_rag --input experiment_data/raw_graph_rag.jsonl --output experiment_data/scored_graph_rag.jsonl
"""

import os
import json
import argparse
import hashlib
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system components
from rag_system.evaluator import AutomatedEvaluator
from rag_system.logger import Logger

def get_deterministic_hash(data_dict: dict) -> str:
    """Generate a consistent MD5 hash across different Python sessions."""
    json_str = json.dumps(data_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generated Questions")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name for isolated logging (e.g., graph_rag)")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to raw JSONL file (e.g., experiment_data/raw_graph_rag.jsonl)")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save scored JSONL file")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

    # ==========================================
    # ISOLATED LOGGING SETUP
    # ==========================================
    log_dir = f"experiment_data/logs_evaluate_{args.model}"
    os.makedirs(log_dir, exist_ok=True)
    eval_logger = Logger(log_dir)

    eval_logger.log(f"[*] Initializing Automated Evaluator for {args.model}...")
    llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    evaluator = AutomatedEvaluator(llm_client)

    if not os.path.exists(args.input):
        eval_logger.log(f"[!] Input file not found: {args.input}")
        return

    # ==========================================
    # CHECKPOINTING & MEMORY RESTORATION LOGIC
    # ==========================================
    scored_questions = set()
    if os.path.exists(args.output):
        eval_logger.log(f"[*] Found existing output '{args.output}'. Loading checkpoints and restoring Evaluator memory...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 1. Register the hash to skip re-evaluating
                    q_dict = data.get("question", {})
                    q_hash = get_deterministic_hash(q_dict)
                    scored_questions.add(q_hash)
                    
                    # 2. Restore Evaluator's "Novelty Penalty" memory state!
                    topic_norm = data.get("metadata", {}).get("topic", "").strip().lower()
                    fmt_label = data.get("metadata", {}).get("question_format", "mcq_single")
                    session_key = (topic_norm, fmt_label)
                    evaluator._session_type_counts[session_key] += 1
                    
                except json.JSONDecodeError:
                    continue
        eval_logger.log(f"[*] Skipping {len(scored_questions)} previously evaluated questions.")

    # ==========================================
    # LOAD PENDING TASKS
    # ==========================================
    pending_tasks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                q_hash = get_deterministic_hash(item)
                if q_hash not in scored_questions:
                    pending_tasks.append(item)
            except json.JSONDecodeError:
                continue

    if not pending_tasks:
        eval_logger.log("[*] All questions in the input file have already been evaluated!")
        return

    eval_logger.log(f"[*] {len(pending_tasks)} questions remaining to be evaluated.")

    # ==========================================
    # EVALUATION LOOP
    # ==========================================
    with open(args.output, 'a', encoding='utf-8') as out_f:
        pbar = tqdm(pending_tasks, desc=f"Evaluating {args.model}", unit="q")
        
        for item in pbar:
            topic = item.get("topic", "Unknown")
            q_format = item.get("question_format", "Unknown")
            
            # Extract the context saved during the generation phase
            saved_context = item.get("retrieved_context", [])
            
            # The evaluator expects the stringified item
            q_json_str = json.dumps(item, ensure_ascii=False)
            
            try:
                # Pass the exactly matched context into the evaluator
                score_result = evaluator.evaluate(question_json=q_json_str, context=saved_context, topic=topic)
                
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
                    "question": item,  # Original raw data + retrieved context
                    "score": score_result
                }
                
                out_f.write(json.dumps(final_item, ensure_ascii=False) + "\n")
                out_f.flush()
                
            except Exception as e:
                eval_logger.log(f"\n[!] Evaluation failed for topic '{topic}': {str(e)}")
                continue

    eval_logger.log(f"\n[*] Evaluation complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()