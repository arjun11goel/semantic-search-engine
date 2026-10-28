"""Shared helpers for the Semantic Search Engine project."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "dataset_embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "embedding_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSIONS = 384


def normalize_text(text: str) -> str:
    """Lowercase text and collapse repeated whitespace for consistent matching."""
    return " ".join(text.strip().lower().split())


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    """Load and validate FAQ records from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset JSON is invalid: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError("The dataset must be a JSON list of FAQ objects.")
    if not records:
        raise ValueError("The dataset is empty. Add between 30 and 50 FAQ entries.")

    required_fields = ("id", "category", "question", "answer")
    seen_ids: set[str] = set()
    validated: list[dict[str, str]] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset entry {index} must be a JSON object.")
        missing = [field for field in required_fields if not isinstance(record.get(field), str) or not record[field].strip()]
        if missing:
            raise ValueError(f"Dataset entry {index} has missing/empty fields: {', '.join(missing)}")

        item = {field: record[field].strip() for field in required_fields}
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate dataset ID found: {item['id']}")
        seen_ids.add(item["id"])
        validated.append(item)

    return validated


def build_embedding_text(record: dict[str, str]) -> str:
    """Build the same compact text representation for indexing and querying."""
    question = normalize_text(record["question"])
    answer = normalize_text(record["answer"])
    category = normalize_text(record["category"])
    return f"question: {question} answer: {answer} category: {category}"


def file_sha256(path: Path) -> str:
    """Return a SHA-256 fingerprint of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between one vector and each row of a matrix."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    rows = np.asarray(matrix, dtype=np.float32)

    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError("The comparison matrix must be one- or two-dimensional.")
    if query.size != rows.shape[1]:
        raise ValueError(
            f"Vector dimensions do not match: query has {query.size}, matrix has {rows.shape[1]}."
        )

    row_norms = np.linalg.norm(rows, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominators = row_norms * query_norm
    numerators = rows @ query
    return np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float32),
        where=denominators > 0,
    )


def load_embedding_model():
    """Load the required local embedding model on CPU with a helpful error message."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers is not installed. Activate the project's virtual "
            "environment and run: python -m pip install -r requirements.txt"
        ) from exc

    print(f"Loading embedding model: {MODEL_NAME}")
    print("Device: CPU. The first run downloads model files and requires internet access.")
    return SentenceTransformer(MODEL_NAME, device="cpu")
