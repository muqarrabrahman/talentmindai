"""Step 3 - Split extracted page text into overlapping, token-based chunks.

Strategy: split each page into sentences, then greedily pack sentences into
~CHUNK_SIZE token chunks with ~CHUNK_OVERLAP tokens of overlap. Chunks never
cross a page boundary, so page_start == page_end and citations stay exact.

Run:  python -m ingestion.chunk
"""
import hashlib
import json
import re

from transformers import AutoTokenizer

import config

# Use the embedding model's own tokenizer so chunk sizes match what BGE will see.
tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_into_sentences(text):
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def overlap_tail(sentences):
    """Keep the last few sentences (up to CHUNK_OVERLAP tokens) for the next chunk."""
    tail, tokens = [], 0
    for sentence in reversed(sentences):
        t = count_tokens(sentence)
        if tokens + t > config.CHUNK_OVERLAP:
            break
        tail.insert(0, sentence)
        tokens += t
    return tail, tokens


def chunk_page(text):
    """Return a list of chunk strings for one page of text."""
    chunks, current, current_tokens = [], [], 0

    for sentence in split_into_sentences(text):
        sent_tokens = count_tokens(sentence)
        if current and current_tokens + sent_tokens > config.CHUNK_SIZE:
            chunks.append(" ".join(current))
            current, current_tokens = overlap_tail(current)
        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc, meta):
    """Turn one extracted document into a list of chunk records."""
    chunks, index = [], 0
    for page in doc["pages"]:
        for text in chunk_page(page["text"]):
            chunks.append({
                "chunk_id": f"{doc['document_id']}_p{page['page_number']:03d}_c{index:03d}",
                "document_id": doc["document_id"],
                "text": text,
                "page_start": page["page_number"],
                "page_end": page["page_number"],
                "chunk_index": index,
                "token_count": count_tokens(text),
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "metadata": {
                    "document_name": meta["document_name"],
                    "source_path": meta["source_path"],
                    "department": meta["department"],
                    "country": meta["country"],
                    "version": meta["version"],
                },
            })
            index += 1
    return chunks


def main():
    config.PROCESSED_CHUNKS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(config.MANIFEST_FILE.read_text())

    all_chunks = []
    for doc_id, meta in manifest.items():
        doc = json.loads((config.PROCESSED_TEXT / f"{doc_id}.json").read_text())
        doc_chunks = chunk_document(doc, meta)
        all_chunks.extend(doc_chunks)
        print(f"{doc_id}: {len(doc_chunks)} chunks")

    out_file = config.PROCESSED_CHUNKS / "chunks.json"
    out_file.write_text(json.dumps(all_chunks, indent=2))
    print(f"Total {len(all_chunks)} chunks -> {out_file}")


if __name__ == "__main__":
    main()
