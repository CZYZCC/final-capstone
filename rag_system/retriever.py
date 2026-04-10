import os
import json
import random
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .knowledge_graph import AdvancedKnowledgeGraph
from rank_bm25 import BM25Okapi
import string


# ---------------------------------------------------------------------------
# Cosine similarity threshold for graph-hop neighbours
# Nodes with cosine(query, node) < this value are dropped from context.
# Tune: lower (0.15) if context is too sparse; raise (0.35) to tighten.
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Helper: encode + normalise a batch of texts for IndexFlatIP
# ---------------------------------------------------------------------------

def _encode_normalised(encoder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(vecs)
    return vecs


# ---------------------------------------------------------------------------
# 1. Pure vector retriever (Baseline)
# ---------------------------------------------------------------------------

class VectorBaselineRetriever:
    """
    Retrieves the top-k most semantically similar KG nodes for a query.

    FIX-1: uses IndexFlatIP (inner product) with L2-normalised embeddings,
    so scores are cosine similarities in [0, 1] instead of Euclidean distances.
    """

    def __init__(self, kg: AdvancedKnowledgeGraph, model_name: str = 'all-MiniLM-L6-v2'):
        self.kg       = kg
        self.encoder  = SentenceTransformer(model_name)
        self.node_ids = list(self.kg.nodes.keys())

        texts      = [self.kg.nodes[nid] for nid in self.node_ids]
        embeddings = _encode_normalised(self.encoder, texts)          # FIX-1
        self.index = faiss.IndexFlatIP(embeddings.shape[1])           # FIX-1
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_vec = _encode_normalised(self.encoder, [query])         # FIX-1
        scores, indices = self.index.search(query_vec, top_k)
        return [
            {
                'node_id': self.node_ids[i],
                'content': self.kg.nodes[self.node_ids[i]],
                'score':   float(scores[0][rank]),
            }
            for rank, i in enumerate(indices[0]) if i != -1
        ]


# ---------------------------------------------------------------------------
# 2. Logic-graph retriever (Hybrid GraphRAG)
# ---------------------------------------------------------------------------

class LogicGraphRetriever:
    """
    Seed retrieval (vector) → hop expansion (graph edges) → cosine rerank.

    FIX-2: final ranked list is filtered by SIMILARITY_THRESHOLD so only
    nodes that are genuinely relevant to the topic reach the generator.
    """

    def __init__(self, kg: AdvancedKnowledgeGraph, vector_retriever: VectorBaselineRetriever):
        self.kg = kg
        self.vector_retriever = vector_retriever

    def retrieve_subgraph(
        self,
        topic: str,
        hops: int = 2,
        max_results: int = 10,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Dict:
        seeds    = self.vector_retriever.retrieve(topic, top_k=5)
        seed_ids = [s['node_id'] for s in seeds]
        self.kg.logger.log(
            f"   [Hybrid-GraphRAG] 以 {len(seed_ids)} 个向量节点为起点逻辑扩散..."
        )

        visited_nodes   = set(seed_ids)
        current         = set(seed_ids)
        extracted_edges = []

        for _ in range(hops):
            next_level = set()
            for node in current:
                neighbours = self.kg.edges.get(node, [])
                neighbours_sorted = sorted(
                    neighbours, key=lambda e: e.get('weight', 1.0), reverse=True
                )[:8]
                for n in neighbours_sorted:
                    next_level.add(n['to'])
                    extracted_edges.append({
                        'subject':   n.get('head_label', node),
                        'predicate': n.get('relation', 'related'),
                        'object':    n.get('tail_label', n['to']),
                    })
            visited_nodes.update(next_level)
            current = next_level

        valid_nodes = [nid for nid in visited_nodes if nid in self.kg.nodes]
        if not valid_nodes:
            return {'nodes': [], 'relations': []}

        # --- Cosine rerank (FIX-1: already using IP index, but here we do
        #     a manual dot product for the hop-expanded nodes not in the index)
        query_vec  = _encode_normalised(self.vector_retriever.encoder, [topic])
        node_texts = [self.kg.nodes[nid] for nid in valid_nodes]
        node_vecs  = _encode_normalised(self.vector_retriever.encoder, node_texts)
        sims       = (node_vecs @ query_vec.T).flatten()

        # FIX-2: apply similarity threshold before ranking
        threshold_pairs = [
            (nid, float(sim))
            for nid, sim in zip(valid_nodes, sims)
            if float(sim) >= similarity_threshold
        ]

        # Fallback: if threshold removes everything, keep top-3 regardless
        if not threshold_pairs:
            self.kg.logger.log(
                f"   [Hybrid-GraphRAG] 阈值 {similarity_threshold} 过滤后无节点，"
                f"回退保留 top-3"
            )
            threshold_pairs = sorted(
                zip(valid_nodes, sims.tolist()), key=lambda x: x[1], reverse=True
            )[:3]

        ranked         = sorted(threshold_pairs, key=lambda x: x[1], reverse=True)[:max_results]
        final_node_ids = {nid for nid, _ in ranked}

        # Deduplicate edges
        seen_edges  = set()
        final_edges = []
        for e in extracted_edges:
            key = (e['subject'], e['predicate'], e['object'])
            if key not in seen_edges:
                seen_edges.add(key)
                final_edges.append(e)
        final_edges = final_edges[:40]

        self.kg.logger.log(
            f"   [Hybrid-GraphRAG] 阈值过滤后保留 {len(ranked)}/{len(valid_nodes)} 个节点，"
            f"{len(final_edges)} 条关系"
        )

        return {
            'nodes':     [{'node_id': nid, 'content': self.kg.nodes[nid]}
                          for nid in final_node_ids],
            'relations': final_edges,
        }


# ---------------------------------------------------------------------------
# 3. Question-bank retriever (Few-shot examples)
# ---------------------------------------------------------------------------

class QuestionBankRetriever:
    """
    Retrieves few-shot examples from question_bank.json.

    FIX-1: same IndexFlatIP + normalisation as VectorBaselineRetriever.
    FIX-3: added retrieve_by_format() to support the 5-format kg_multitype bank.
    """

    def __init__(self, question_bank_path: str, encoder: SentenceTransformer):
        self.encoder   = encoder
        self.questions: List[Dict] = []
        self.index     = None

        if not os.path.exists(question_bank_path):
            print(f"[!] 找不到题库文件 {question_bank_path}，Few-shot 功能将被禁用。")
            return

        with open(question_bank_path, 'r', encoding='utf-8') as f:
            self.questions = json.load(f)

        if self.questions:
            # Use 'question' field; fall back to 'statement' (true_false) or 'sentence' (fill_blank)
            texts = [
                q.get('question') or q.get('statement') or q.get('sentence') or ''
                for q in self.questions
            ]
            embeddings = _encode_normalised(self.encoder, texts)       # FIX-1
            self.index = faiss.IndexFlatIP(embeddings.shape[1])        # FIX-1
            self.index.add(embeddings)

            comp     = sum(1 for q in self.questions if q.get('type') == 'computational'
                           or q.get('question_type') == 'computational')
            mt_count = sum(1 for q in self.questions if q.get('source') == 'kg_multitype')
            print(
                f"[*] 题库已加载：{len(self.questions)} 道题"
                f"（计算 {comp} / 多题型 {mt_count}）"
            )

    @property
    def available(self) -> bool:
        return self.index is not None and len(self.questions) > 0

    # ------------------------------------------------------------------
    # Core retrieval (used by SmartGenerator for reuse/vary)
    # ------------------------------------------------------------------

    def retrieve_similar(
        self,
        topic: str,
        top_k: int = 3,
        prefer_computational: bool = True,
        question_format: Optional[str] = None,   # FIX-3: format filter
    ) -> List[Dict]:
        """
        Retrieve top-k most similar questions to topic.

        question_format: if given, filter candidates to that format only
                         (e.g. 'mcq_single', 'true_false').
                         None means no format filter (original behaviour).
        """
        if not self.available:
            return []

        query_vec    = _encode_normalised(self.encoder, [topic])
        k_candidates = min(top_k * 10, len(self.questions))
        scores, indices = self.index.search(query_vec, k_candidates)
        candidates   = [self.questions[i] for i in indices[0] if i != -1]

        # FIX-3: optional format filter
        if question_format is not None:
            candidates = [
                q for q in candidates
                if q.get('question_format') == question_format
            ]

        if prefer_computational and question_format is None:
            comp   = [q for q in candidates
                      if q.get('type') == 'computational'
                      or q.get('question_type') == 'computational']
            concep = [q for q in candidates
                      if q.get('type') != 'computational'
                      and q.get('question_type') != 'computational']
            selected = (comp[:2] + concep[:max(0, top_k - 2)])[:top_k]
            if len(selected) < top_k:
                selected = candidates[:top_k]
        else:
            selected = candidates[:top_k]

        return selected

    # ------------------------------------------------------------------
    # FIX-3: format-specific retrieval for the 5-type question bank
    # ------------------------------------------------------------------

    def retrieve_by_format(
        self,
        topic: str,
        question_format: str,
        top_k: int = 2,
    ) -> List[Dict]:
        """
        Retrieve examples of a specific question format for a topic.
        Used by SmartGenerator to build few-shot context for each format.

        question_format: one of 'mcq_single', 'mcq_multi', 'true_false',
                         'fill_blank', 'open_answer'
        """
        return self.retrieve_similar(
            topic,
            top_k=top_k,
            prefer_computational=False,
            question_format=question_format,
        )

    # ------------------------------------------------------------------
    # Helpers used by SmartGenerator._find_best_question
    # ------------------------------------------------------------------

    def get_computational_questions(self, topic: str, top_k: int = 5) -> List[Dict]:
        if not self.available:
            return []

        query_vec    = _encode_normalised(self.encoder, [topic])
        k_candidates = min(top_k * 10, len(self.questions))
        scores, indices = self.index.search(query_vec, k_candidates)
        candidates   = [self.questions[i] for i in indices[0] if i != -1]
        return [
            q for q in candidates
            if q.get('type') == 'computational'
            or q.get('question_type') == 'computational'
        ][:top_k]
    

    # ----- 将以下代码添加到 retriever.py 的最下方 -----

# ---------------------------------------------------------------------------
# 4. Hybrid Retriever (Vector + BM25 + RRF) - 新增 Baseline C
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    结合 Dense Vector 和 Sparse BM25 的混合检索器。
    使用 RRF (Reciprocal Rank Fusion) 算法进行结果重排。
    """
    def __init__(self, vector_retriever: VectorBaselineRetriever):
        self.vector_retriever = vector_retriever
        self.kg = vector_retriever.kg
        self.node_ids = vector_retriever.node_ids
        
        # 构建 BM25 索引 (简单分词：去除标点并按空格切分)
        corpus = []
        for nid in self.node_ids:
            text = self.kg.nodes[nid].lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            corpus.append(text.split())
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
        # 1. 获取向量召回结果 (召回两倍数量用于重排)
        vec_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        
        # 2. 获取 BM25 召回结果
        query_tokens = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        
        # 3. RRF (Reciprocal Rank Fusion) 分数计算
        rrf_scores = {}
        
        # 处理 Vector 排名
        for rank, res in enumerate(vec_results):
            nid = res['node_id']
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # 处理 BM25 排名
        for rank, idx in enumerate(bm25_indices):
            nid = self.node_ids[idx]
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # 4. 根据 RRF 分数重排并返回 Top-K
        sorted_nids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        return [
            {
                'node_id': nid,
                'content': self.kg.nodes[nid],
                'score': rrf_scores[nid]
            }
            for nid in sorted_nids
        ]
