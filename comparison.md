# Keyword Search vs. Semantic Search

## Purpose

This experiment compares embedding-based semantic search with a deliberately simple keyword-overlap baseline using the same 40 student tech-support FAQ records. Five fixed queries are passed to both methods.

Semantic search embeds the query and ranks FAQ records by cosine similarity against the cached dataset vectors. Keyword search tokenizes the query, removes common stop words, and ranks records by the fraction of remaining query tokens that also occur in the FAQ question, answer, or category. The keyword baseline does not perform stemming, synonym expansion, or contextual language understanding.

## Test queries and recorded top results

The following table records the output from the completed local run on Windows using `sentence-transformers/all-MiniLM-L6-v2` and the included dataset. Scores can vary somewhat with model/library versions; they are ranking signals, not probabilities.

<!-- BEGIN GENERATED RESULTS -->
| # | Query | Semantic top result | Cosine similarity | Keyword top result | Assessment |
|---:|---|---|---:|---|---|
| 1 | My notebook goes dead before the lecture ends. | FAQ025 — How do I share my screen during an online class? | 0.2884 | No match | Top semantic result was incorrect. The interactive top-three search did include the battery-life FAQ (FAQ011), but only at rank 3. |
| 2 | My handset cannot browse while I am indoors. | FAQ009 — Why does a website work on mobile data but not on Wi-Fi? | 0.3825 | No match | Partially relevant to connectivity, although the result does not fully resolve the query's ambiguity about indoor browsing. |
| 3 | I made a mistake and now the earlier material is gone. | FAQ018 — How can I recover edits from a document that closed without saving? | 0.3146 | FAQ025 — How do I share my screen during an online class? (16.7% overlap; shared token: `material`) | Semantic search found a relevant recovery FAQ; keyword search returned a false positive caused by a shared word. FAQ017, about restoring a deleted file, was the second semantic result in the interactive search. |
| 4 | The machine is scorching and no longer responds. | FAQ021 — Why is my printer not printing anything? | 0.3572 | FAQ005 — How can I sign out of my account on another device? (25.0% overlap; shared token: `longer`) | Both top results were poor. The query combines overheating and unresponsiveness, and the system failed to prioritize the relevant device-overheating FAQ. |
| 5 | Booting is painfully sluggish, and the desktop takes forever to appear. | FAQ013 — Why does my computer start slowly? | 0.5732 | No match | Semantic search retrieved the intended startup-performance FAQ, while the keyword baseline found no shared non-stopword terms. |
<!-- END GENERATED RESULTS -->

## Findings

### Where semantic search helped

**Query 3 — document recovery:** The query does not use the FAQ wording “recover edits” or “closed without saving.” Semantic search placed FAQ018 first, with a cosine similarity of `0.3146`, and the interactive top-three output also returned FAQ017, which covers restoring a deleted file. The keyword baseline instead selected FAQ025 because the word `material` appeared in its online-class content. This example shows how word overlap can be distracted by a shared term while semantic retrieval can locate a conceptually related record.

**Query 5 — slow startup:** The query describes a sluggish boot process without using the exact wording “start slowly.” Semantic search ranked FAQ013 first with a similarity of `0.5732`. Keyword search returned no match because it had no non-stopword terms in common with the dataset text. This was the clearest positive example in the five-query run.

### Partial result

**Query 2 — indoor connectivity:** Semantic search returned FAQ009 about a website working on mobile data but not Wi-Fi, and keyword search returned no match. This is related to the connectivity theme, but it is not a fully precise answer to the phrase “cannot browse while I am indoors.” It should be counted as partially relevant rather than a clear retrieval success.

### Where semantic search did not work well

**Query 1 — notebook battery:** The top semantic result was FAQ025 about screen sharing during an online class, which is not the requested battery issue. The interactive top-three output ranked the intended battery-life entry, FAQ011, third. This means the relevant record was present in the returned set but was not ranked highly enough.

**Query 4 — overheating and unresponsiveness:** Semantic search ranked FAQ021 about a printer not printing, while the keyword baseline ranked FAQ005 about signing out of an account. Neither top result addressed the combined symptoms. The FAQ dataset contains an overheating question, but the current model did not place it first for this phrasing. This is a limitation of the current retrieval setup, not evidence that the query was successfully answered.

## Conclusion

For these five manually inspected queries, semantic search produced a clearly relevant top result in **two cases**, a partially relevant result in **one case**, and a poor top result in **two cases**. Keyword search missed useful paraphrases for queries 2 and 5, and a shared token misled it on query 3. However, semantic search also made ranking mistakes on queries 1 and 4.

The experiment supports a balanced conclusion: semantic embeddings can help retrieve information expressed with different words, but they do not guarantee relevance. The small dataset, the combination of question/answer/category text in each embedding, and the limited five-query sample all affect the findings. A larger evaluation set with expected relevant FAQ IDs would be needed for a meaningful retrieval-quality metric.

## Reproducing the comparison

After installing the dependencies and generating the cache, run:

```powershell
python compare_search.py
```

The script runs the same five queries against both methods, prints the top result for each, and updates the generated-results section above with the current top-three rankings and keyword matches. The interpretive findings below the generated section record the observed run documented in this report.
