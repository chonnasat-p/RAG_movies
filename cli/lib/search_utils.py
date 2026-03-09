import json
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies():
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopswords():
    with open("data/stopwords.txt", 'r') as file:
        return file.read().splitlines()

def format_search_result(
    doc_id: str, title: str, document: str, score: float) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
    }