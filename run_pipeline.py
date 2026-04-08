import time
from rag_system import Pipeline
from dotenv import load_dotenv
import os
# 加载环境变量
load_dotenv()

# ==========================================
# CONFIG
# ==========================================
API_KEY            = os.getenv("DEEPSEEK_API_KEY")
TEXTBOOK_DIR       = "./GraphRAG-Bench/textbooks"
TRIPLETS_PATH      = "./global_knowledge_graph.json"
QUESTION_BANK_PATH = "./question_bank.json"

# TOPICS = [
#     "recursion",
#     "sorting algorithm",
#     "graph traversal",
#     "dynamic programming",
#     "hash table",
#     "Cybersecurity"
# ]
TOPICS = [

    "Cybersecurity",
    "cloud computing",
    "recursion"
]


# ==========================================


def main():
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./experiment_logs/run_{timestamp}"

    pipeline = Pipeline(
        api_key=API_KEY,
        output_dir=output_dir,
        question_bank_path=QUESTION_BANK_PATH,
    )

    pipeline.run(
        textbook_dir  =TEXTBOOK_DIR,
        triplets_path =TRIPLETS_PATH,
        topics        =TOPICS,
    )


if __name__ == "__main__":
    main()
