"""Demonstrate sentence embeddings and cosine similarity."""

from __future__ import annotations

import numpy as np

from utils import cosine_similarity, load_embedding_model, normalize_text


SAMPLE_SENTENCES = [
    "The laptop battery dies very quickly.",
    "My notebook loses power after a short time.",
    "The computer has poor battery life.",
    "Water boils at one hundred degrees Celsius.",
]


def main() -> None:
    model = load_embedding_model()
    sentences = [normalize_text(sentence) for sentence in SAMPLE_SENTENCES]
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("\nEMBEDDING BASICS DEMO")
    print("=" * 64)
    for index, sentence in enumerate(SAMPLE_SENTENCES, start=1):
        print(f"{index}. {sentence}")
    print(f"\nNumber of sample sentences: {len(sentences)}")
    print(f"Embedding array shape: {embeddings.shape}")
    print(f"Each sentence vector has {embeddings.shape[1]} numbers.")

    comparisons = [
        (0, 1, "related: laptop battery vs. notebook losing power"),
        (0, 2, "related: laptop battery vs. computer battery life"),
        (0, 3, "unrelated: laptop battery vs. boiling water"),
    ]
    print("\nCOSINE SIMILARITY COMPARISONS")
    print("=" * 64)
    for left, right, label in comparisons:
        score = float(cosine_similarity(embeddings[left], embeddings[right : right + 1])[0])
        print(f"{label}\n  score: {score:.4f}")

    related_score = float(cosine_similarity(embeddings[0], embeddings[1:2])[0])
    unrelated_score = float(cosine_similarity(embeddings[0], embeddings[3:4])[0])
    print("\nSanity check:")
    if related_score > unrelated_score:
        print("PASS: the related example scores higher than the unrelated example.")
    else:
        print("NOTICE: this example did not follow the expected ordering; inspect the scores above.")
    print("Similarity scores are relative measures, not probabilities.")


if __name__ == "__main__":
    main()
