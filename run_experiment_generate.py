"""
run_experiment_generate.py
Large-Scale Question Generation Script with Checkpointing and Isolated Logging.
Usage:
    python run_experiment_generate.py --model no_retrieval
    python run_experiment_generate.py --model vector_rag
    python run_experiment_generate.py --model graph_rag
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system components
from rag_system.knowledge_graph import AdvancedKnowledgeGraph
from rag_system.retriever import VectorBaselineRetriever, LogicGraphRetriever, QuestionBankRetriever
from rag_system.generator import SmartGenerator, BaselineGenerator, NoRetrievalGenerator
from rag_system.logger import Logger

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
TEXTBOOK_DIR       = "./GraphRAG-Bench/textbooks"
TRIPLETS_PATH      = "./global_knowledge_graph.json"
QUESTION_BANK_PATH = "./question_bank.json"

# Exactly matching the topics in your evaluator.py
COMPUTATIONAL_TOPICS = [
    "hash table", "merge sort", "quick sort", "heap sort", 
    "bfs graph traversal", "dfs graph traversal", "dynamic programming knapsack", 
    "recursion", "binary search tree", "dijkstra shortest path", 
    "binary search", "hash table linear probing", "data structures and algorithms"
]

CONCEPTUAL_TOPICS = [
    "cybersecurity", "computer networks", "information retrieval", 
    "machine learning", "operating systems", "database systems", "human-computer interaction"
]

TOPICS = COMPUTATIONAL_TOPICS + CONCEPTUAL_TOPICS
FORMATS = ["mcq_single", "mcq_multi", "true_false", "fill_blank", "open_answer"]
QUESTIONS_PER_COMBO = 4  # 20 topics * 5 formats * 4 questions = 400 total questions per model

def initialize_components(model_name: str):
    """Initialize Knowledge Graph, Retrievers, Logger, and API key with isolation."""
    print(f"[*] Loading knowledge graph and retrievers for {model_name}...")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

    # 1. Initialize Logger with Isolated Directory
    log_dir = f"experiment_data/logs_generate_{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    experiment_logger = Logger(log_dir)

    # 2. Initialize Knowledge Graph
    kg = AdvancedKnowledgeGraph(logger=experiment_logger)
    
    if os.path.exists(TEXTBOOK_DIR):
        kg.build_base_structure(TEXTBOOK_DIR)
    else:
        print(f"[!] Warning: TEXTBOOK_DIR '{TEXTBOOK_DIR}' not found. Base nodes may be empty.")

    if os.path.exists(TRIPLETS_PATH):
        kg.load_triplets(TRIPLETS_PATH)
    else:
        print(f"[!] Warning: TRIPLETS_PATH '{TRIPLETS_PATH}' not found.")
    
    # 3. Initialize Retrievers
    vector_retriever = VectorBaselineRetriever(kg)
    logic_retriever = LogicGraphRetriever(kg, vector_retriever)
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    qb_retriever = QuestionBankRetriever(QUESTION_BANK_PATH, encoder)
    
    return api_key, kg, vector_retriever, logic_retriever, qb_retriever

def main():
    parser = argparse.ArgumentParser(description="Large-Scale Question Generation")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["no_retrieval", "vector_rag", "graph_rag"], 
                        help="Target generation model")
    args = parser.parse_args()

    # Initialize system with the specific model name to isolate logs
    api_key, kg, vector_retriever, logic_retriever, qb_retriever = initialize_components(args.model)
    
    # Initialize the appropriate generator
    print(f"[*] Initializing {args.model} generator...")
    if args.model == "no_retrieval":
        generator = NoRetrievalGenerator(api_key=api_key)
    elif args.model == "vector_rag":
        generator = BaselineGenerator(api_key=api_key)
    elif args.model == "graph_rag":
        generator = SmartGenerator(api_key=api_key)

    output_file = f"experiment_data/raw_{args.model}.jsonl"

    # ==========================================
    # CHECKPOINTING / RESUMPTION LOGIC
    # ==========================================
    completed_counts = {(t, f): 0 for t in TOPICS for f in FORMATS}

    if os.path.exists(output_file):
        print(f"[*] Found existing dataset '{output_file}'. Scanning for checkpoints...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    t = data.get("topic")
                    fmt = data.get("question_format")
                    if (t, fmt) in completed_counts:
                        completed_counts[(t, fmt)] += 1
                except json.JSONDecodeError:
                    continue
        
        total_done = sum(completed_counts.values())
        print(f"[*] Resuming experiment. {total_done} questions already generated.")

    # ==========================================
    # MAIN GENERATION LOOP
    # ==========================================
    total_tasks = len(TOPICS) * len(FORMATS) * QUESTIONS_PER_COMBO
    current_done = sum(completed_counts.values())
    
    if current_done >= total_tasks:
        print(f"[*] Generation for {args.model} is already 100% complete.")
        return

    print(f"[*] Starting generation loop for {args.model}...")
    with open(output_file, 'a', encoding='utf-8') as out_f:
        pbar = tqdm(total=total_tasks, initial=current_done, desc=f"Generating {args.model}")

        for topic in TOPICS:
            for q_format in FORMATS:
                needed = QUESTIONS_PER_COMBO - completed_counts[(topic, q_format)]
                
                for _ in range(needed):
                    start_time = time.time()
                    try:
                        saved_context = []
                        q_type = "computational" if topic in COMPUTATIONAL_TOPICS else "conceptual"

                        # Retrieve context based on the model and save it for the evaluator
                        if args.model == "no_retrieval":
                            raw_json, method = generator.generate(
                                topic=topic, 
                                qb_retriever=qb_retriever, question_format=q_format
                            )
                        elif args.model == "vector_rag":
                            context = vector_retriever.retrieve(topic, top_k=5)
                            saved_context = context
                            raw_json, method = generator.generate(
                                topic=topic, context=context, 
                                qb_retriever=qb_retriever, question_format=q_format
                            )
                        elif args.model == "graph_rag":
                            graph_ctx = logic_retriever.retrieve_subgraph(topic, hops=2)
                            saved_context = graph_ctx
                            raw_json, method = generator.generate(
                                topic=topic, graph_context=graph_ctx, 
                                qb_retriever=qb_retriever, question_format=q_format
                            )
                        
                        # Parse and annotate the result
                        q_data = json.loads(raw_json)
                        q_data["topic"] = topic
                        q_data["model"] = args.model
                        q_data["method"] = method
                        q_data["generation_time"] = round(time.time() - start_time, 2)
                        q_data["question_format"] = q_format
                        q_data["expected_type"] = q_type
                        q_data["retrieved_context"] = saved_context  # <-- Crucial for evaluation
                        
                        # Save to disk immediately
                        out_f.write(json.dumps(q_data, ensure_ascii=False) + "\n")
                        out_f.flush() 
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n[!] Error generating -> Topic: {topic} | Format: {q_format} | Error: {str(e)}")
                        print("[*] Sleeping for 5 seconds to prevent rate limit blocks...")
                        time.sleep(5) 

    pbar.close()
    print(f"\n[*] Generation completed successfully! Data saved to: {output_file}")

if __name__ == "__main__":
    main()