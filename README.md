# SmartQG: Logical Graph Retrieval-Augmented Generation For Multi-type Question Construction

SmartQG is an advanced framework designed for the automated construction of high-quality, psychometrically sound Computer Science (CS) exam questions. Developed as a Capstone Project under the supervision of Prof. HUANG Xiao, the system addresses the limitations of naive RAG (Chunking Fragmentation) and traditional GraphRAG (Neighbor Noise).

## 🚀 Overview
SmartQG utilizes an **Executable Algorithm Knowledge Graph (GEAKG)** and a **Seed-and-Prune Logical Graph Retrieval** mechanism to force LLMs to synthesize cross-concept relationships and trigger adversarial edge cases.

## ✨ Core Contributions
1.  **Executable Algorithm Knowledge Graph (GEAKG)**:
    * Contains **4,659 nodes** and **37,998 relational triplets**.
    * Encodes five typed procedural relations: `HAS_STEP`, `TRUE_BRANCH`, `FALSE_BRANCH`, `PRODUCES_OUTPUT`, and `HAS_COMPLEXITY`.
2.  **Seed-and-Prune Retrieval with Topic-Aware Adaptation**:
    * Implements **Topic-Aware Pruning**: Dynamically adjusts similarity thresholds (0.25 for computational vs. 0.10 for conceptual) and subgraph expansion limits to maximize context relevance.
    * Eliminates neighbor noise by pruning nodes below the adaptive cosine similarity threshold.
3.  **Multi-Format Generation & Self-Correction**:
    * Supports 5 formats: **MCQ Single, MCQ Multi, True/False, Fill-in-Blank, and Open Answer**.
    * Guarded by a **Format-Aware Self-Correction Loop**: Uses tailored difficulty thresholds (e.g., threshold 1 for Fill-in-Blank to preserve accuracy, threshold 3 for others) to ensure university-level rigor.

## 📂 Repository Structure
* `build_kg.py`: KG extraction from textbooks and question bank construction.
* `run_all.py`: Overnight runner orchestrating the evaluation of **7 model configurations** across 2,100 questions.
* `generator.py`: Specialized generators (No-Retrieval, Vector-RAG, Smart-GraphRAG) with difficulty-driven self-correction.
* `retriever.py`: Multi-modal search implementing Vector search, Topic-Adaptive Logic Graph expansion, and Hybrid RRF.
* `evaluator.py`: Automated psychometric judge scoring across 7 dimensions (Relevance, Correctness, Diagnostic Power, Multi-Hop, Edge-Case, Graph Depth, Diversity).

## 🛠️ Installation & Usage

### Setup
1. Install dependencies:
   `pip install openai sentence-transformers faiss-cpu rank_bm25 python-dotenv tqdm`
2. Configure `.env` with your `DEEPSEEK_API_KEY`.
3. Place data in `./GraphRAG-Bench/textbooks` and `./GraphRAG-Bench/questions`.

### Execution
**Build Knowledge Graph:**
`python build_kg.py --mode all`

**Run Evaluation Pipeline:**
`python run_all.py`

## 🎓 Evaluation Dimensions
* Relevance (5%) | Diversity (10%) | Correctness (20%) | Diagnostic Power (20%) | Multi-Hop Dependency (15%) | Edge-Case Triggering (20%) | Graph-Relational Depth (10%).
