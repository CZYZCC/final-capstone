# Single-Input / Single-Output Question Generation Pipeline

This README explains the latest modification of the project. The update focuses on three goals:

1. Replace the original benchmark-dataset-driven workflow with a **single-input / single-output** workflow.
2. Standardize both input and output as **JSON**.
3. Separate **knowledge graph construction** from **question generation**, so that an already-built KG can be reused as the input of question generation.

The original benchmark scripts are still kept for legacy experiments, including `run_all.py`, `run_experiment_generate.py`, and `run_experiment_evaluate.py`. For the new single-question experiment, use `single_io.py`.

---

## 1. What Changed

### 1.1 Previous benchmark-style workflow

The original project was mainly designed for full benchmark experiments:

```text
benchmark textbooks / benchmark dataset
        ↓
build a global knowledge graph
        ↓
iterate over TOPICS × QUESTION_FORMATS
        ↓
generate many questions
        ↓
evaluate and save batch results
```

This is useful for large-scale experiments, but it is inconvenient when we only want to test one topic, one KG, and one question format.

### 1.2 New single-input / single-output workflow

The new workflow is:

```text
topic + KG reference + question_format
        ↓
single_io.py --mode generate
        ↓
one JSON result containing question + answer + metadata
```

The KG can be provided in three ways:

1. Inline `knowledge_graph` in the request JSON.
2. A local KG file path using `kg_path`.
3. A reusable KG identifier using `kg_id`, which is resolved from `kg_store/<kg_id>.json`.

This avoids repeatedly copying a full KG into every generation request.

---

## 2. Modified and Added Files

| File | Purpose |
|---|---|
| `single_io.py` | New command-line entry point for single JSON input and single JSON output. It supports `generate` and `build_kg` modes. |
| `rag_system/json_io.py` | JSON API layer. It validates the request, resolves KG input, normalizes KG format, calls `SmartGenerator`, and returns a stable JSON response. |
| `build_kg.py` | Adds `LLMGraphExtractor.build_from_texts()`, allowing KG construction from JSON-provided text instead of scanning the benchmark dataset. |
| `rag_system/generator.py` | Supports explicit `question_type` and single-question generation without a benchmark question-bank retriever. |
| `kg_store/` | New folder for storing reusable KG JSON files. |
| `examples/generate_request.json` | Example request with an inline KG. |
| `examples/generate_with_kg_path_request.json` | Example request that references a KG file by `kg_path`. |
| `examples/generate_with_kg_id_request.json` | Example request that references a KG file by `kg_id`. |
| `examples/build_kg_request.json` | Example request for building a KG from raw text. |

---

## 3. Environment Setup

Activate your Python environment first. For example, if you use the `pytorch` conda environment:

```bash
conda activate pytorch
```

Install required packages if they are not already installed:

```bash
pip install openai python-dotenv tqdm sentence-transformers faiss-cpu rank_bm25
```

Create a `.env` file in the project root:

```bash
printf "DEEPSEEK_API_KEY=your_deepseek_api_key\n" > .env
```

Do not commit `.env` to GitHub. It contains a private API key.

---

## 4. Mode 1: Generate One Question from a KG

### 4.1 Recommended usage: use `kg_path`

This is the most practical format for local experiments. The request only contains the topic, question format, and the path to a saved KG file.

Example file: `examples/generate_with_kg_path_request.json`

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "question_type": "computational",
  "kg_path": "kg_store/binary_search.json"
}
```

Run:

```bash
mkdir -p outputs
python single_io.py \
  --mode generate \
  --input examples/generate_with_kg_path_request.json \
  --output outputs/question_output.json
```

View the output:

```bash
python -m json.tool outputs/question_output.json
```

---

### 4.2 Shorter usage: use `kg_id`

If a KG file is saved as:

```text
kg_store/binary_search.json
```

then the request can use:

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "question_type": "computational",
  "kg_id": "binary_search"
}
```

Run:

```bash
python single_io.py \
  --mode generate \
  --input examples/generate_with_kg_id_request.json \
  --output outputs/question_output.json
```

By default, `kg_id` is resolved from:

```text
kg_store/<kg_id>.json
```

For example:

```text
kg_id = binary_search
resolved file = kg_store/binary_search.json
```

You can customize the KG store directory with an environment variable:

```bash
export SMARTQG_KG_STORE=my_kg_store
```

or by putting `kg_store_dir` into the request JSON:

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "kg_id": "binary_search",
  "kg_store_dir": "kg_store"
}
```

---

### 4.3 Backward-compatible usage: inline `knowledge_graph`

The old single-input format is still supported. This is useful for small tests, but it becomes verbose when the KG is large.

Example file: `examples/generate_request.json`

```json
{
  "topic": "hash table linear probing",
  "question_format": "mcq_single",
  "question_type": "computational",
  "knowledge_graph": {
    "nodes": [
      {
        "node_id": "n1",
        "content": "Linear probing resolves a hash collision by checking the next table slot until an empty slot is found."
      }
    ],
    "relations": [
      {
        "subject": "collision at initial slot",
        "predicate": "HAS_STEP",
        "object": "linear probe to next available slot"
      }
    ]
  }
}
```

Run:

```bash
python single_io.py \
  --mode generate \
  --input examples/generate_request.json \
  --output outputs/question_output.json
```

---

## 5. Mode 2: Build One KG from Raw Text

KG construction and question generation are now separated. You can build a KG once, save it, and reuse it many times.

Example file: `examples/build_kg_request.json`

```json
{
  "topic": "binary search",
  "kg_id": "binary_search_generated",
  "texts": [
    {
      "node_id": "bs_1",
      "content": "Binary search compares the target with the middle element of a sorted array. If the target is smaller than the middle element, the algorithm continues on the left half. If the target is larger, it continues on the right half. Binary search has time complexity O(log n)."
    }
  ]
}
```

Build the KG and save it into the KG store:

```bash
mkdir -p kg_store
python single_io.py \
  --mode build_kg \
  --input examples/build_kg_request.json \
  --output kg_store/binary_search_generated.json
```

The output format is:

```json
{
  "topic": "binary search",
  "kg_id": "binary_search_generated",
  "knowledge_graph": {
    "nodes": [],
    "relations": [],
    "triplets": []
  },
  "metadata": {
    "mode": "single_kg_build",
    "input_texts": 1,
    "kg_nodes": 1,
    "kg_relations": 4,
    "kg_triplets": 4
  }
}
```

After building the KG, generate a question with `kg_id`:

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "question_type": "computational",
  "kg_id": "binary_search_generated"
}
```

Then run:

```bash
python single_io.py \
  --mode generate \
  --input your_generate_request.json \
  --output outputs/question_output.json
```

---

## 6. Input JSON Schema for Question Generation

The generation request must contain:

| Field | Required | Description |
|---|---:|---|
| `topic` | Yes | The topic of the question, e.g., `binary search`. |
| `question_format` | Yes | The desired question format. |
| `question_type` | No | `computational` or `conceptual`. If omitted, the generator can use its internal logic. |
| `knowledge_graph` | Conditional | Inline KG object. Use this only for small examples. |
| `kg_path` | Conditional | Path to a saved KG JSON file. |
| `kg_id` | Conditional | KG identifier resolved from `kg_store/<kg_id>.json`. |
| `kg_store_dir` | No | Custom KG store directory for resolving `kg_id`. |
| `naive_mode` | No | Optional boolean flag used by the generator. |
| `no_self_correction` | No | Optional boolean flag used by the generator. |

Exactly one of the following KG inputs is usually enough:

```text
knowledge_graph
kg_path
kg_id
```

If more than one is provided, the priority is:

```text
knowledge_graph > kg_path > kg_id
```

---

## 7. Supported Question Formats

| `question_format` | Meaning | Answer field extracted into `response["answer"]` |
|---|---|---|
| `mcq_single` | Single-choice question | `correct_answer` |
| `mcq_multi` | Multiple-choice question | `correct_answers` |
| `true_false` | True/false question | `tf_answer` |
| `fill_blank` | Fill-in-the-blank question | `answers` |
| `open_answer` | Open-ended question | `model_answer` |

The helper function `extract_answer()` in `rag_system/json_io.py` converts different answer fields into one stable output field:

```python
response["answer"]
```

---

## 8. Output JSON Schema

A successful generation output looks like this:

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "question_type": "computational",
  "question": {
    "question": "...",
    "correct_answer": "...",
    "rationale": "...",
    "question_format": "mcq_single",
    "question_type": "computational"
  },
  "answer": "...",
  "metadata": {
    "method": "generated",
    "model": "graph_rag_single_io",
    "generation_time": 12.34,
    "kg_nodes": 4,
    "kg_relations": 4,
    "kg_source": "kg_id",
    "kg_id": "binary_search",
    "kg_path": "kg_store/binary_search.json"
  }
}
```

Important fields:

| Output field | Meaning |
|---|---|
| `question` | Full raw question object generated by the model. |
| `answer` | A stable answer field extracted from the raw question object. |
| `metadata.kg_source` | Shows whether the KG came from `inline`, `kg_path`, or `kg_id`. |
| `metadata.kg_nodes` | Number of normalized KG nodes used by the generator. |
| `metadata.kg_relations` | Number of normalized KG relations used by the generator. |

---

## 9. Python API Usage

You can also call the new single-question API directly in Python:

```python
from rag_system.json_io import generate_question_from_json

request = {
    "topic": "binary search",
    "question_format": "mcq_single",
    "question_type": "computational",
    "kg_id": "binary_search",
}

response = generate_question_from_json(request)
print(response["question"])
print(response["answer"])
```

For KG construction:

```python
from single_io import build_kg_from_json

request = {
    "topic": "binary search",
    "texts": [
        {
            "node_id": "bs_1",
            "content": "Binary search compares the target with the middle element of a sorted array."
        }
    ]
}

kg_response = build_kg_from_json(request)
print(kg_response["knowledge_graph"])
```

---

## 10. Common Commands

Generate from inline KG:

```bash
python single_io.py --mode generate \
  --input examples/generate_request.json \
  --output outputs/question_output.json
```

Generate from KG path:

```bash
python single_io.py --mode generate \
  --input examples/generate_with_kg_path_request.json \
  --output outputs/question_output.json
```

Generate from KG id:

```bash
python single_io.py --mode generate \
  --input examples/generate_with_kg_id_request.json \
  --output outputs/question_output.json
```

Build KG from raw text:

```bash
python single_io.py --mode build_kg \
  --input examples/build_kg_request.json \
  --output kg_store/binary_search_generated.json
```

Pretty-print output:

```bash
python -m json.tool outputs/question_output.json
```

---

## 11. Troubleshooting

### 11.1 `DEEPSEEK_API_KEY is required`

The `.env` file is missing or does not contain the key.

Fix:

```bash
printf "DEEPSEEK_API_KEY=your_deepseek_api_key\n" > .env
```

### 11.2 `Authentication Fails, Your api key is invalid`

The code successfully loaded an API key, but the DeepSeek server rejected it. This is not a JSON or KG-format issue. Regenerate the API key and update `.env`.

### 11.3 `KG id 'xxx' was not found`

The request uses:

```json
{"kg_id": "xxx"}
```

but the file does not exist at:

```text
kg_store/xxx.json
```

Fix by either creating the file or using `kg_path` directly.

### 11.4 `Knowledge graph file not found`

The `kg_path` value is wrong. Check that the file exists:

```bash
ls kg_store/binary_search.json
```

### 11.5 `question_format must be one of ...`

Use one of:

```text
mcq_single
mcq_multi
true_false
fill_blank
open_answer
```

---

## 12. Recommended Final Workflow

For real use, the recommended workflow is:

```text
Step 1: Build KG once
raw text → build_kg → kg_store/<kg_id>.json

Step 2: Reuse KG many times
{topic, question_format, kg_id} → generate → one question + answer JSON
```

This design keeps the teacher's required separation between KG construction and question generation, while avoiding the redundancy of copying a full KG into every input request.
