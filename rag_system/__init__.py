"""rag_system package exports with optional heavy dependencies."""

from .logger          import Logger
from .knowledge_graph import AdvancedKnowledgeGraph
from .json_io         import generate_question_from_json, normalize_graph_context, extract_answer

try:
    from .generator import NoRetrievalGenerator, BaselineGenerator, SmartGenerator
except ModuleNotFoundError:
    NoRetrievalGenerator = BaselineGenerator = SmartGenerator = None

try:
    from .retriever import VectorBaselineRetriever, LogicGraphRetriever, QuestionBankRetriever
except ModuleNotFoundError:
    VectorBaselineRetriever = LogicGraphRetriever = QuestionBankRetriever = None

try:
    from .evaluator import AutomatedEvaluator
except ModuleNotFoundError:
    AutomatedEvaluator = None

try:
    from .pipeline import Pipeline
except ModuleNotFoundError:
    Pipeline = None
