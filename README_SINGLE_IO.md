# Single JSON Input / Output 修改说明

本 README 专门说明本项目最近围绕以下三点所做的修改：

1. 将原先基于 benchmark dataset 的批量处理方式，调整为 **单条输入 / 单条输出** 的形式。
2. 将输入和输出统一为 **JSON 格式**。
3. 将 **knowledge graph 构建** 与 **question generation** 拆分开，使已经生成好的 KG 可以直接作为问题生成函数的输入。

> 说明：原来的 benchmark 脚本仍然保留，例如 `run_all.py`、`run_experiment_generate.py`、`run_experiment_evaluate.py`。它们主要用于旧的批量实验。现在如果要进行单条实验，推荐使用新的入口 `single_io.py` 或 Python 函数 `generate_question_from_json()`。

---

## 1. 修改后的整体流程

### 原始 benchmark 流程

原来的代码主要围绕固定 benchmark dataset 运行，大致流程是：

```text
benchmark dataset / textbooks
        ↓
批量构建 KG
        ↓
按 TOPICS × FORMATS 批量生成问题
        ↓
批量评估与保存结果
```

这种方式适合跑完整实验，但不方便对单个 topic、单个 KG、单个 question format 做独立测试。

### 新的单条输入 / 单条输出流程

现在新增了一个单条 JSON 调用流程：

```text
topic + knowledge_graph + question_format
        ↓
single_io.py --mode generate
        ↓
一个 JSON 输出：question + answer + metadata
```

也就是说，现在你可以只输入一个 topic、一个已经构建好的 KG、一个题型，然后直接得到一个问题和答案。

---

## 2. 本次新增和修改的核心文件

| 文件 | 作用 |
|---|---|
| `single_io.py` | 新增命令行入口，负责读取单个 JSON 输入文件，并输出单个 JSON 结果。支持 `generate` 和 `build_kg` 两种模式。 |
| `rag_system/json_io.py` | 新增 JSON API 层，负责校验输入 JSON、规范化 KG 格式、调用生成器、统一输出 JSON。 |
| `build_kg.py` | 新增 `LLMGraphExtractor.build_from_texts()`，让 KG 构建可以从单条 JSON 文本输入开始，而不是只能扫描 benchmark 文件夹。 |
| `rag_system/generator.py` | 修改 `SmartGenerator.generate()`，允许外部显式传入 `question_type`，并支持在没有 benchmark question bank 的情况下生成单条问题。 |
| `examples/generate_request.json` | 新增示例：使用已经构建好的 KG 生成一道题。 |
| `examples/build_kg_request.json` | 新增示例：从一段或多段文本构建 KG。 |

---

## 3. 需求一：单条输入 / 单条输出

### 3.1 单条问题生成入口

现在单条问题生成不需要跑 `run_all.py`，而是运行：

```bash
python single_io.py --mode generate --input examples/generate_request.json --output question_output.json
```

其中：

- `--mode generate` 表示执行 question generation。
- `--input` 是单条 JSON 输入文件。
- `--output` 是单条 JSON 输出文件。

如果不写 `--output`，程序也会直接把 JSON 输出打印在终端中。

---

### 3.2 单条输入 JSON 格式

最小输入格式如下：

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

字段说明：

| 字段 | 是否必需 | 含义 |
|---|---:|---|
| `topic` | 必需 | 当前要生成问题的主题，例如 `binary search`、`hash table linear probing`。 |
| `question_format` | 必需 | 要生成的问题类型，例如单选题、多选题、判断题、填空题、开放问答。 |
| `question_type` | 可选 | 问题认知类型，目前支持 `computational` 和 `conceptual`。如果不传，会沿用原代码中的自动判断逻辑。 |
| `knowledge_graph` | 必需 | 已经构建好的 KG，作为后续 question generation 的输入。 |

---

### 3.3 支持的问题格式

当前支持以下 `question_format`：

| question_format | 含义 | 输出答案字段 |
|---|---|---|
| `mcq_single` | 单选题 | `correct_answer` |
| `mcq_multi` | 多选题 | `correct_answers` |
| `true_false` | 判断题 | `tf_answer` |
| `fill_blank` | 填空题 | `answers` |
| `open_answer` | 开放问答题 | `model_answer` |

`rag_system/json_io.py` 里的 `extract_answer()` 会根据不同题型自动抽取答案，并统一放到输出 JSON 的 `answer` 字段中。

---

### 3.4 单条输出 JSON 格式

运行完成后，输出大致如下：

```json
{
  "topic": "hash table linear probing",
  "question_format": "mcq_single",
  "question_type": "computational",
  "question": {
    "question": "...",
    "correct_answer": "...",
    "rationale": "...",
    "distractors": [
      {
        "option": "...",
        "explanation": "..."
      }
    ],
    "question_format": "mcq_single",
    "question_type": "computational"
  },
  "answer": "...",
  "metadata": {
    "method": "generated",
    "model": "graph_rag_single_io",
    "generation_time": 12.34,
    "kg_nodes": 1,
    "kg_relations": 1
  }
}
```

字段说明：

| 输出字段 | 含义 |
|---|---|
| `topic` | 本次输入的主题。 |
| `question_format` | 本次生成的问题格式。 |
| `question_type` | 本次生成的问题类型。 |
| `question` | 模型生成的完整问题对象。 |
| `answer` | 从 `question` 中抽取出的统一答案字段，方便后续评测或展示。 |
| `metadata` | 运行信息，例如生成方法、耗时、KG 节点数、KG 关系数。 |

这样做的好处是：无论原始题型是什么，外部调用者都可以稳定地读取：

```python
response["question"]
response["answer"]
```

---

## 4. 需求二：输入和输出统一为 JSON

本次修改后，命令行入口 `single_io.py` 只接收 JSON 文件，并且只输出 JSON 结果。

### 4.1 读取 JSON

`single_io.py` 内部通过 `_read_json()` 读取输入：

```python
request = _read_json(args.input)
```

它要求输入文件必须是一个 JSON object，也就是 Python 里的 `dict`，不能是纯数组或普通字符串。

正确：

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "knowledge_graph": {
    "nodes": [],
    "relations": []
  }
}
```

错误：

```json
[
  {"topic": "binary search"}
]
```

因为这是数组，不是单条 JSON object。

---

### 4.2 写出 JSON

`single_io.py` 内部通过 `_write_json()` 输出结果：

```python
_write_json(response, args.output)
```

如果指定 `--output`，结果会保存到文件；同时也会打印到终端。输出使用：

```python
json.dumps(data, ensure_ascii=False, indent=2)
```

因此中文内容不会被转义成 Unicode 编码，便于直接阅读。

---

## 5. 需求三：KG 构建与问题生成拆分

本次修改后，KG 构建和问题生成被拆成两个独立步骤。

### 5.1 步骤 A：只构建 KG

如果你还没有 KG，可以先从文本构建 KG：

```bash
python single_io.py --mode build_kg --input examples/build_kg_request.json --output kg_output.json
```

输入示例：

```json
{
  "topic": "binary search",
  "texts": [
    {
      "node_id": "bs_1",
      "content": "Binary search compares the target with the middle element of a sorted array. If the target is smaller, it continues on the left half. If the target is larger, it continues on the right half. The time complexity is O(log n)."
    }
  ]
}
```

也可以使用单个 `text` 字段：

```json
{
  "topic": "binary search",
  "text": "Binary search compares the target with the middle element of a sorted array..."
}
```

输出示例：

```json
{
  "topic": "binary search",
  "knowledge_graph": {
    "nodes": [
      {
        "node_id": "bs_1",
        "content": "Binary search compares the target with the middle element of a sorted array..."
      }
    ],
    "relations": [
      {
        "subject": "target smaller than middle element",
        "predicate": "TRUE_BRANCH",
        "object": "continues on the left half",
        "source_node": "bs_1"
      }
    ],
    "triplets": [
      {
        "head": "target smaller than middle element",
        "relation": "TRUE_BRANCH",
        "tail": "continues on the left half",
        "source_node": "bs_1"
      }
    ]
  },
  "metadata": {
    "mode": "single_kg_build",
    "input_texts": 1,
    "kg_nodes": 1,
    "kg_relations": 1,
    "kg_triplets": 1
  }
}
```

---

### 5.2 步骤 B：把已经生成好的 KG 作为输入，生成问题

拿到上一步的 `knowledge_graph` 后，可以把它放进 generate 输入中：

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "question_type": "computational",
  "knowledge_graph": {
    "nodes": [
      {
        "node_id": "bs_1",
        "content": "Binary search compares the target with the middle element of a sorted array..."
      }
    ],
    "relations": [
      {
        "subject": "target smaller than middle element",
        "predicate": "TRUE_BRANCH",
        "object": "continues on the left half"
      }
    ]
  }
}
```

然后运行：

```bash
python single_io.py --mode generate --input generate_request.json --output question_output.json
```

因此，新的逻辑已经变成：

```text
KG 构建函数：文本 → knowledge_graph JSON
问题生成函数：topic + knowledge_graph + question_format → question + answer JSON
```

这正好对应“已经生成好的 KG 是 question generation 函数的 input”。

---

## 6. Python 函数调用方式

除了命令行，也可以在 Python 中直接调用。

### 6.1 直接基于已有 KG 生成问题

```python
from rag_system.json_io import generate_question_from_json

request = {
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

response = generate_question_from_json(request)
print(response["question"])
print(response["answer"])
```

---

### 6.2 从文本构建 KG

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

## 7. KG 输入格式的兼容性

`rag_system/json_io.py` 中的 `normalize_graph_context()` 支持多种 KG 写法。

### 7.1 nodes + relations

```json
{
  "nodes": [
    {
      "node_id": "n1",
      "content": "..."
    }
  ],
  "relations": [
    {
      "subject": "A",
      "predicate": "HAS_STEP",
      "object": "B"
    }
  ]
}
```

### 7.2 triplets

```json
{
  "triplets": [
    {
      "head": "A",
      "relation": "HAS_STEP",
      "tail": "B"
    }
  ]
}
```

### 7.3 edges

```json
{
  "edges": [
    {
      "from": "A",
      "type": "HAS_STEP",
      "to": "B"
    }
  ]
}
```

内部会统一转成：

```json
{
  "nodes": [...],
  "relations": [
    {
      "subject": "...",
      "predicate": "...",
      "object": "..."
    }
  ]
}
```

---

## 8. 环境变量设置

两个模式都会调用 DeepSeek API，因此需要设置 `DEEPSEEK_API_KEY`。

推荐方式是在本地环境变量中设置：

```bash
export DEEPSEEK_API_KEY="你的 key"
```

或者在项目根目录创建 `.env`：

```bash
DEEPSEEK_API_KEY=你的 key
```

注意：不要把 `.env` 上传到 GitHub，也不要提交给老师或他人，因为里面可能包含 API key。

---

## 9. 推荐的单条实验顺序

### 情况一：你已经有 KG

直接运行：

```bash
python single_io.py --mode generate --input examples/generate_request.json --output question_output.json
```

流程为：

```text
已有 KG
  ↓
generate
  ↓
question_output.json
```

---

### 情况二：你还没有 KG

先构建 KG：

```bash
python single_io.py --mode build_kg --input examples/build_kg_request.json --output kg_output.json
```

再把 `kg_output.json` 中的 `knowledge_graph` 放入 generate 输入，生成问题：

```bash
python single_io.py --mode generate --input generate_request.json --output question_output.json
```

流程为：

```text
原始文本
  ↓
build_kg
  ↓
knowledge_graph
  ↓
generate
  ↓
question_output.json
```

---

## 10. 与旧 benchmark 代码的关系

旧代码没有删除，主要是为了保留原始实验能力：

| 旧文件 | 现在的作用 |
|---|---|
| `run_all.py` | 仍然可以用于完整批量实验，但不是单条实验入口。 |
| `run_experiment_generate.py` | 仍然是 benchmark topic / format 批量生成逻辑。 |
| `run_experiment_evaluate.py` | 仍然用于原来的批量评估。 |
| `global_knowledge_graph.json` | 旧流程或已有全局 KG 文件。单条实验可以不用它。 |
| `question_bank.json` | 旧流程的问题库。单条输入可以不依赖它。 |

新的主入口是：

```text
single_io.py
```

新的 JSON API 层是：

```text
rag_system/json_io.py
```

因此，如果你的目标是验证这三条修改是否实现，应该优先看：

```text
single_io.py
rag_system/json_io.py
build_kg.py 中的 build_from_texts()
rag_system/generator.py 中 SmartGenerator.generate() 的 question_type 参数
```

---

## 11. 常见错误

### 错误 1：没有设置 API key

报错类似：

```text
DEEPSEEK_API_KEY is required, either in environment or as api_key.
```

解决：

```bash
export DEEPSEEK_API_KEY="你的 key"
```

---

### 错误 2：输入 JSON 不是 object

如果输入是数组：

```json
[
  {"topic": "binary search"}
]
```

会报错，因为现在是单条输入 / 单条输出，所以最外层必须是一个 JSON object：

```json
{
  "topic": "binary search",
  "question_format": "mcq_single",
  "knowledge_graph": {
    "nodes": [],
    "relations": []
  }
}
```

---

### 错误 3：缺少 knowledge_graph

报错类似：

```text
request.knowledge_graph is required.
```

解决：在 generate 输入中加入：

```json
"knowledge_graph": {
  "nodes": [...],
  "relations": [...]
}
```

---

### 错误 4：question_format 写错

目前支持：

```text
mcq_single
mcq_multi
true_false
fill_blank
open_answer
```

如果写成 `multiple_choice`、`single_choice` 等，会被拒绝。

---

## 12. 最终总结

本次修改后的代码满足三条目标：

1. **单条输入 / 单条输出**：通过 `single_io.py --mode generate` 实现，输入一个 JSON，输出一个 JSON。
2. **输入输出统一 JSON**：`single_io.py` 只读 JSON object，输出统一结构的 JSON response。
3. **KG 构建与问题生成拆分**：`build_kg` 模式只负责构建 KG；`generate` 模式把已有 KG 作为输入生成问题和答案。

因此，现在项目既保留了原来的 benchmark 批量实验能力，也支持更清晰的单条实验接口。
