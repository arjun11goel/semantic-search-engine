"""Create and cache embeddings for the FAQ dataset."""

from __future__ import annotations

import json

import numpy as np

from utils import (
    DATASET_PATH,
    EMBEDDINGS_DIR,
    EMBEDDINGS_PATH,
    EXPECTED_EMBEDDING_DIMENSIONS,
    METADATA_PATH,
    MODEL_NAME,
    build_embedding_text,
    file_sha256,
    load_dataset,
    load_embedding_model,
)


def main() -> None:
    records = load_dataset()
    if not 30 <= len(records) <= 50:
        raise ValueError(
            f"The task requires 30–50 entries; the current dataset contains {len(records)}."
        )

    texts = [build_embedding_text(record) for record in records]
    print(f"Dataset: {DATASET_PATH}")
    print(f"Validated FAQ records: {len(records)}")

    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (len(records), EXPECTED_EMBEDDING_DIMENSIONS):
        raise RuntimeError(
            "Unexpected embedding shape. Expected "
            f"({len(records)}, {EXPECTED_EMBEDDING_DIMENSIONS}), got {embeddings.shape}."
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=False)

    metadata = {
        "model_name": MODEL_NAME,
        "dataset_file": "data/dataset.json",
        "dataset_sha256": file_sha256(DATASET_PATH),
        "embedding_shape": list(embeddings.shape),
        "embedding_dimensions": int(embeddings.shape[1]),
        "normalized_embeddings": True,
        "text_fields": ["question", "answer", "category"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("\nEMBEDDINGS CREATED SUCCESSFULLY")
    print(f"Saved vectors: {EMBEDDINGS_PATH.relative_to(EMBEDDINGS_PATH.parents[1])}")
    print(f"Saved metadata: {METADATA_PATH.relative_to(METADATA_PATH.parents[1])}")
    print(f"Embedding shape: {embeddings.shape}")
    print("Dataset embeddings are cached. Search queries will not re-embed the dataset.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"\nERROR: {exc}") from exc
