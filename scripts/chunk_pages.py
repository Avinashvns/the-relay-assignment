import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "paper_pages.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SETTINGS = {
    "small_k3": {
        "chunk_words": 180,
        "overlap_words": 30,
        "top_k": 3,
        "rerank": False,
    },
    "medium_k4": {
        "chunk_words": 320,
        "overlap_words": 50,
        "top_k": 4,
        "rerank": False,
    },
    "large_k5_rerank": {
        "chunk_words": 500,
        "overlap_words": 80,
        "top_k": 5,
        "rerank": True,
    },
}


def read_pages():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Page data file not found: {INPUT_FILE}")

    with INPUT_FILE.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def split_page(page_record, setting_name, config):
    words = page_record["text"].split()
    chunk_words = config["chunk_words"]
    overlap_words = config["overlap_words"]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "chunk_id": (
                f"{setting_name}-{page_record['paper_id']}"
                f"-p{page_record['page']}-c{chunk_index:03d}"
            ),
            "setting": setting_name,
            "chunk_index": chunk_index,
            "paper_id": page_record["paper_id"],
            "title": page_record["title"],
            "version": page_record["version"],
            "page": page_record["page"],
            "start_word": start,
            "end_word": end,
            "word_count": end - start,
            "top_k": config["top_k"],
            "rerank": config["rerank"],
            "text": chunk_text,
        })

        if end == len(words):
            break

        start = end - overlap_words
        chunk_index += 1

    return chunks


def main():
    pages = read_pages()

    if not pages:
        raise ValueError("The paper pages file is empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for setting_name, config in SETTINGS.items():
        all_chunks = []

        for page_record in pages:
            all_chunks.extend(
                split_page(page_record, setting_name, config)
            )

        output_file = OUTPUT_DIR / f"chunks_{setting_name}.jsonl"

        with output_file.open("w", encoding="utf-8") as file:
            for chunk in all_chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(
            f"{setting_name}: "
            f"{len(all_chunks)} pieces -> {output_file.name}"
        )


if __name__ == "__main__":
    main()