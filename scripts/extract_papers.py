import json
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "paper_pages.jsonl"

PAPERS = {
    "attention_is_all_you_need_v7.pdf": {
        "paper_id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "version": "v7",
    },
    "moshi_v1.pdf": {
        "paper_id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "version": "v1",
    },
    "dynamic_chunking_v2.pdf": {
        "paper_id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "version": "v2",
    },
}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    page_records = []

    for filename, metadata in PAPERS.items():
        pdf_path = PAPERS_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF Not Found: {pdf_path}")

        with pymupdf.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if not text:
                    continue

                page_records.append({
                    **metadata,
                    "page": page_index + 1,
                    "text": text,
                })

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for record in page_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Recevied pages: {len(page_records)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()