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

__all__ = [
    "cosine_similarity",
    "generate_query_embedding",
    "rank_chunks_by_similarity",
    "search_similar_chunks",
]


if __name__ == "__main__":
    from src.similarity_demo import run_similarity_demonstration
    run_similarity_demonstration()

