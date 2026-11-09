import json
import math
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "how", "in", "into", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text):
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS
    ]


class BM25Retriever:
    def __init__(self):
        self.corpora = {}
        self._load_corpora()

    def _load_corpora(self):
        chunk_files = sorted(CHUNKS_DIR.glob("chunks_*.jsonl"))

        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk files found in: {CHUNKS_DIR}"
            )

        for chunk_file in chunk_files:
            setting_name = chunk_file.stem.removeprefix("chunks_")
            chunks = []

            with chunk_file.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    chunk = json.loads(line)
                    tokens = tokenize(chunk["text"])
                    chunk["_tokens"] = tokens
                    chunk["_term_frequency"] = Counter(tokens)
                    chunk["_token_count"] = len(tokens)
                    chunks.append(chunk)

            if not chunks:
                raise ValueError(f"Chunk file is empty: {chunk_file}")

            document_frequency = Counter()

            for chunk in chunks:
                document_frequency.update(set(chunk["_tokens"]))

            document_count = len(chunks)
            inverse_document_frequency = {
                term: math.log(
                    1 + (document_count - frequency + 0.5)
                    / (frequency + 0.5)
                )
                for term, frequency in document_frequency.items()
            }

            average_document_length = (
                sum(chunk["_token_count"] for chunk in chunks)
                / document_count
            )

            self.corpora[setting_name] = {
                "chunks": chunks,
                "top_k": chunks[0]["top_k"],
                "rerank": chunks[0]["rerank"],
                "idf": inverse_document_frequency,
                "average_length": average_document_length,
            }

    def _bm25_score(self, query_terms, chunk, idf, average_length):
        score = 0.0
        term_frequency = chunk["_term_frequency"]
        document_length = chunk["_token_count"]

        for term in query_terms:
            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            denominator = frequency + BM25_K1 * (
                1 - BM25_B
                + BM25_B * document_length / average_length
            )

            score += (
                idf.get(term, 0.0)
                * frequency
                * (BM25_K1 + 1)
                / denominator
            )

        return score

    def retrieve(self, query, setting_name):
        if setting_name not in self.corpora:
            available = ", ".join(sorted(self.corpora))
            raise ValueError(
                f"Unknown setting '{setting_name}'. Available: {available}"
            )

        query_terms = set(tokenize(query))

        if not query_terms:
            raise ValueError("The query has no searchable terms.")

        corpus = self.corpora[setting_name]
        chunks = corpus["chunks"]
        top_k = corpus["top_k"]
        rerank_enabled = corpus["rerank"]
        idf = corpus["idf"]
        average_length = corpus["average_length"]

        scored_chunks = []

        for chunk in chunks:
            score = self._bm25_score(
                query_terms,
                chunk,
                idf,
                average_length,
            )
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        if not rerank_enabled:
            return [
                self._format_result(rank, score, None, chunk)
                for rank, (score, chunk) in enumerate(
                    scored_chunks[:top_k],
                    start=1,
                )
            ]

        candidate_count = min(len(scored_chunks), top_k * 3)
        candidates = scored_chunks[:candidate_count]

        candidate_scores = [score for score, _ in candidates]
        minimum_score = min(candidate_scores)
        maximum_score = max(candidate_scores)
        score_range = maximum_score - minimum_score

        reranked = []

        for bm25_score, chunk in candidates:
            if score_range == 0:
                normalized_bm25 = 0.0
            else:
                normalized_bm25 = (
                    bm25_score - minimum_score
                ) / score_range

            chunk_terms = set(chunk["_tokens"])
            term_coverage = (
                len(query_terms & chunk_terms) / len(query_terms)
            )

            title_terms = set(tokenize(chunk["title"]))
            title_match = (
                len(query_terms & title_terms) / len(query_terms)
            )

            rerank_score = (
                0.70 * normalized_bm25
                + 0.25 * term_coverage
                + 0.05 * title_match
            )

            reranked.append(
                (rerank_score, bm25_score, chunk)
            )

        reranked.sort(key=lambda item: item[0], reverse=True)

        return [
            self._format_result(rank, bm25_score, rerank_score, chunk)
            for rank, (rerank_score, bm25_score, chunk)
            in enumerate(reranked[:top_k], start=1)
        ]

    @staticmethod
    def _format_result(rank, bm25_score, rerank_score, chunk):
        return {
            "rank": rank,
            "paper_id": chunk["paper_id"],
            "title": chunk["title"],
            "version": chunk["version"],
            "page": chunk["page"],
            "bm25_score": round(bm25_score, 4),
            "rerank_score": (
                round(rerank_score, 4)
                if rerank_score is not None
                else None
            ),
            "text": chunk["text"],
        }


def main():
    retriever = BM25Retriever()
    available_settings = ", ".join(sorted(retriever.corpora))

    print(f"Available settings: {available_settings}")
    setting_name = input("Enter a setting: ").strip()
    query = input("Enter a question: ").strip()

    results = retriever.retrieve(query, setting_name)

    for result in results:
        citation = (
            f"[{result['paper_id']}, page {result['page']}]"
        )

        print(f"\n{result['rank']}. {citation} {result['title']}")
        print(f"BM25 score: {result['bm25_score']}")

        if result["rerank_score"] is not None:
            print(f"Rerank score: {result['rerank_score']}")

        print(result["text"][:700])


if __name__ == "__main__":
    main()