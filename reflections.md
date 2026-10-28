# Week 2 Reflection — Embeddings and Semantic Search

## What is an embedding?

An embedding is a list of numbers that represents text in a form that a machine-learning system can compare. Instead of checking only whether two sentences contain the same words, an embedding model maps text into a numerical vector space in which sentences with related meanings can often be closer together. In this project, the `sentence-transformers/all-MiniLM-L6-v2` model converts each FAQ record into a 384-dimensional vector. I compare a query vector with the cached FAQ vectors using cosine similarity and return the records with the highest scores.

The embedding does not preserve a perfect definition of a sentence, and a similarity score does not guarantee that a retrieved answer is correct. It is a ranking signal. The returned question and answer still need to be inspected for relevance.

## What I observed in the embedding experiment

The embedding demo produced an array of shape `(4, 384)`, confirming that four sample sentences were represented by 384-number vectors. In my run, the laptop-battery sentence had a cosine similarity of `0.6369` with a sentence about a notebook losing power, and `0.6692` with a sentence about poor computer battery life. Its similarity with the unrelated boiling-water sentence was `0.0024`. These values showed the expected pattern for these examples: the related sentences scored much higher than the unrelated one.

I also embedded all 40 FAQ records and obtained a matrix of shape `(40, 384)`. This confirms that the full dataset was embedded and that the saved vector matrix has one row per FAQ.

## An example where semantic search beat keyword search

The clearest example was the query: **“Booting is painfully sluggish, and the desktop takes forever to appear.”** The semantic search engine returned FAQ013, “Why does my computer start slowly?”, as the top result with a cosine similarity of `0.5732`. Keyword search returned no match because the query did not share any non-stopword terms with the FAQ text.

Another useful example was “I made a mistake and now the earlier material is gone.” Semantic search returned FAQ018, which explains how to recover edits from a document closed without saving. The keyword baseline instead returned FAQ025 about screen sharing because both texts included the word `material`. This showed me that a word-overlap search can be distracted by a shared word even when the overall topics are different.

## What the comparison also taught me

Semantic search did not get every query right. For “My notebook goes dead before the lecture ends,” the top result was about screen sharing, although the battery-life FAQ appeared at rank three in the interactive results. For “The machine is scorching and no longer responds,” the top result was about a printer not printing. These examples showed me why a high-ranked embedding result must not automatically be treated as the correct answer.

In the five-query comparison, I judged two semantic top results clearly relevant, one partially relevant, and two poor. This is only a small manual experiment and not a general accuracy measurement. The dataset contains 40 entries and the test has only five queries, so more examples and manually labelled expected results would be needed to evaluate retrieval quality more systematically.

## Why cache the embeddings?

The FAQ dataset does not change every time a user submits a query, so recomputing all its vectors on every search would waste time. `embed_data.py` embeds the dataset and saves the vectors in `embeddings/dataset_embeddings.npy`. It also writes metadata containing the model name, embedding dimensions, and a SHA-256 fingerprint of the dataset. During a query, the program embeds only the new query and compares it against the cached matrix. The metadata check detects when the dataset has changed and the cache should be rebuilt.

## What are the limits of the in-memory approach at larger scale?

This implementation loads the vector matrix into RAM and compares a query with every stored vector. That is simple and suitable for a 40-entry dataset. As the number of documents and vector dimensions grow, the matrix consumes more memory and each query requires more comparisons. A larger application may also need reliable updates, metadata filters, persistence, concurrent access, and a retrieval index instead of a full linear scan.

## How could ChromaDB help?

ChromaDB is a vector database that can store embeddings along with associated documents and metadata. Its client supports local persistent storage, allowing a collection to be saved on disk instead of managing only a NumPy matrix in application code. A vector database can also provide collection-management and metadata-filtering features and can help organize retrieval as the dataset expands. A local Chroma setup can be explored without paying for a hosted database.

I have not added ChromaDB to this Week 2 implementation because the task specifically focuses on generating embeddings, caching a `.npy` matrix, and implementing cosine-similarity search directly. Using NumPy keeps those operations visible and helps me understand the retrieval process before adding a vector database in a later stage.

## References

- [Model card: sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — describes the model as producing 384-dimensional dense vectors for semantic similarity and retrieval tasks.
- [Sentence Transformers documentation](https://sbert.net/docs/installation.html) — installation and usage information.
- [Chroma documentation](https://docs.trychroma.com/docs/overview/getting-started) — usage guidance for storing and querying data with Chroma.
- [Chroma client documentation](https://cookbook.chromadb.dev/core/clients/) — local persistent-client options and storage on the local machine.

## Conclusion

This project helped me understand the basic retrieval workflow used in a RAG system: represent documents as embeddings, cache those vectors, embed the incoming query, calculate cosine similarity, and retrieve relevant records. The results also showed me that semantic search is not automatically accurate. The next improvement would be to refine the FAQ entries and test a larger set of paraphrases with expected relevant IDs, then compare top-k retrieval results systematically.
