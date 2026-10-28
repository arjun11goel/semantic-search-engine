"""Interactive CLI for semantic FAQ search and a keyword-search baseline."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from utils import (
    DATASET_PATH,
    EMBEDDINGS_PATH,
    EXPECTED_EMBEDDING_DIMENSIONS,
    METADATA_PATH,
    MODEL_NAME,
    build_embedding_text,
    cosine_similarity,
    file_sha256,
    load_dataset,
    load_embedding_model,
    normalize_text,
)


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "being",
    "by", "can", "could", "did", "do", "does", "for", "from", "get", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "please", "should", "so", "that", "the", "their", "them",
    "there", "this", "to", "was", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "not", "no", "t",
    "s", "re", "ll", "ve", "d", "m", "am", "have", "has", "had", "cannot",
    "cant", "cannot", "very", "much", "just", "than", "then", "after", "again",
}


def tokenize(text: str) -> set[str]:
    """Tokenize normalized text for the deliberately simple keyword baseline."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if token not in STOP_WORDS and len(token) > 1
    }


def keyword_search(
    query: str,
    records: list[dict[str, str]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rank FAQ entries by the fraction of query tokens that overlap the FAQ text."""
    if not query.strip() or top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    matches: list[dict[str, Any]] = []
    for record in records:
        record_tokens = tokenize(build_embedding_text(record))
        overlap = query_tokens.intersection(record_tokens)
        if not overlap:
            continue
        result: dict[str, Any] = dict(record)
        result["score"] = len(overlap) / len(query_tokens)
        result["matched_terms"] = sorted(overlap)
        matches.append(result)

    matches.sort(key=lambda item: (-item["score"], -len(item["matched_terms"]), item["id"]))
    return matches[:top_k]


class SemanticSearchEngine:
    """Load cached FAQ vectors and retrieve entries by cosine similarity."""

    def __init__(self) -> None:
        self.records = load_dataset()
        if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Cached dataset embeddings were not found. Run `python embed_data.py` first."
            )

        try:
            import json

            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read embedding metadata: {exc}") from exc

        if metadata.get("model_name") != MODEL_NAME:
            raise RuntimeError("The cached model name does not match. Run `python embed_data.py` again.")
        current_hash = file_sha256(DATASET_PATH)
        if metadata.get("dataset_sha256") != current_hash:
            raise RuntimeError(
                "The dataset has changed since its embeddings were created. "
                "Run `python embed_data.py` to refresh the cache."
            )

        try:
            self.embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load cached embeddings: {exc}") from exc

        expected_shape = (len(self.records), EXPECTED_EMBEDDING_DIMENSIONS)
        if self.embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Cached vector shape is {self.embeddings.shape}; expected {expected_shape}. "
                "Run `python embed_data.py` again."
            )
        if not np.isfinite(self.embeddings).all():
            raise RuntimeError("The cached embeddings contain invalid numeric values. Rebuild the cache.")

        self.model = None

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the top-k FAQ entries sorted by cosine similarity to the query."""
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.model is None:
            self.model = load_embedding_model()

        query_embedding = self.model.encode(
            [normalize_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)[0]
        scores = cosine_similarity(query_embedding, self.embeddings)
        best_indices = np.argsort(scores)[::-1][: min(top_k, len(self.records))]

        results: list[dict[str, Any]] = []
        for index in best_indices:
            item: dict[str, Any] = dict(self.records[int(index)])
            item["score"] = float(scores[int(index)])
            results.append(item)
        return results


def print_semantic_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No results found.")
        return
    print("\nTop semantic matches (cosine similarity; not a probability):")
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. [{result['id']}] {result['question']}")
        print(f"   Category: {result['category']} | Similarity: {result['score']:.4f}")
        print(f"   Answer: {result['answer']}")


def main() -> None:
    try:
        engine = SemanticSearchEngine()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"\nSETUP ERROR: {exc}") from exc

    print("\nSEMANTIC SEARCH ENGINE")
    print("=" * 64)
    print(f"Loaded {len(engine.records)} FAQ entries and cached vectors.")
    print("Type a question to search. Commands: /help, /quit, /exit")

    try:
        while True:
            try:
                query = input("\nSearch> ").strip()
            except EOFError:
                print("\nInput ended. Goodbye!")
                break

            if not query:
                print("Please enter a non-empty search query.")
                continue
            command = query.lower()
            if command in {"/quit", "/exit", "quit", "exit", "0"}:
                print("Goodbye!")
                break
            if command == "/help":
                print("Enter a natural-language question to retrieve the top 3 FAQ entries.")
                print("Commands: /help, /quit, /exit")
                continue

            try:
                print_semantic_results(engine.search(query, top_k=3))
            except (RuntimeError, ValueError) as exc:
                print(f"Search error: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
