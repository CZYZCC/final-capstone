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

## Single JSON I/O Mode

The original benchmark mode is still available, but production-style usage should call the single JSON I/O API.

### 1. Build a KG JSON from one request

```bash
python single_io.py --mode build_kg \
  --input examples/build_kg_request.json \
  --output kg_output.json
```

Input:
```json
{
  "topic": "hash table linear probing",
  "texts": [
    {"node_id": "hash_1", "content": "...source knowledge text..."}
  ]
}
```

Output:
```json
{
  "topic": "hash table linear probing",
  "knowledge_graph": {
    "nodes": [{"node_id": "hash_1", "content": "..."}],
    "relations": [{"subject": "...", "predicate": "HAS_STEP", "object": "..."}],
    "triplets": [{"head": "...", "relation": "HAS_STEP", "tail": "...", "source_node": "hash_1"}]
  },
  "metadata": {"mode": "single_kg_build"}
}
```

### 2. Generate one question from an already-built KG

```bash
python single_io.py --mode generate \
  --input examples/generate_request.json \
  --output question_output.json
```

Input:
```json
{
  "topic": "hash table linear probing",
  "question_format": "mcq_single",
  "question_type": "computational",
  "knowledge_graph": {
    "nodes": [{"node_id": "n1", "content": "..."}],
    "relations": [{"subject": "...", "predicate": "HAS_STEP", "object": "..."}]
  }
}
```

Output:
```json
{
  "topic": "hash table linear probing",
  "question_format": "mcq_single",
  "question_type": "computational",
  "question": {"question": "...", "correct_answer": "..."},
  "answer": "...",
  "metadata": {"model": "graph_rag_single_io", "kg_nodes": 1, "kg_relations": 1}
}
```


## Latest Single-I/O Update

The latest single-input/single-output pipeline is documented in [`README_SINGLE_IO.md`](README_SINGLE_IO.md). It supports three KG input styles for single-question generation: inline `knowledge_graph`, file-based `kg_path`, and reusable `kg_id` resolved from `kg_store/<kg_id>.json`.
