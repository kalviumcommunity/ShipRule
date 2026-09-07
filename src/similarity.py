import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embeddings import (
    cosine_similarity,
    generate_query_embedding,
    rank_chunks_by_similarity,
    search_similar_chunks,
)
from src.retrieval import (
    retrieve,
    keyword_score,
    hybrid_rank,
    hybrid_retrieve,
    format_retrieval_output,
    format_filtered_vs_unfiltered_output,
    format_hybrid_output,
    load_indexed_vector_collection,
    run_retrieval_demonstration,
)
from src.reranker import (
    rerank_score,
    rerank,
    retrieve_and_rerank,
    format_before_after_comparison,
    format_rank_movement_report,
)
from src.evaluation import (
    evaluate_retrieval,
    compare_configurations,
    format_manual_inspection_view,
    EVALUATION_DATASET,
    EVALUATION_CONFIGURATIONS,
)

__all__ = [
    "cosine_similarity",
    "generate_query_embedding",
    "rank_chunks_by_similarity",
    "search_similar_chunks",
    "retrieve",
    "keyword_score",
    "hybrid_rank",
    "hybrid_retrieve",
    "format_retrieval_output",
    "format_filtered_vs_unfiltered_output",
    "format_hybrid_output",
    "load_indexed_vector_collection",
    "run_retrieval_demonstration",
    "rerank_score",
    "rerank",
    "retrieve_and_rerank",
    "format_before_after_comparison",
    "format_rank_movement_report",
    "evaluate_retrieval",
    "compare_configurations",
    "format_manual_inspection_view",
    "EVALUATION_DATASET",
    "EVALUATION_CONFIGURATIONS",
]



if __name__ == "__main__":
    run_retrieval_demonstration()
