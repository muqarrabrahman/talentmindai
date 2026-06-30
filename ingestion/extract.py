"""Step 2 - Extract per-page text from policy PDFs.

Reads PDFs from data/raw/policies/ and writes one JSON file per document to
data/processed/policies_text/. Page numbers are preserved so we can cite them later.

Run:  python -m ingestion.extract
"""
import json

import fitz  # PyMuPDF

import config


def extract_pdf(path):
    """Return [{"page_number": int, "text": str}, ...] for one PDF."""
    pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:                                  # skip blank/scanned-image pages
            pages.append({"page_number": i, "text": text})
    doc.close()
    return pages


def build_manifest(pdf_paths):
    """Assign each PDF a stable document_id (policy_001, ...) plus editable metadata.

    Edit the generated data/processed/manifest.json to set the real
    department / country / version for each document.
    """
    manifest = {}
    for idx, path in enumerate(sorted(pdf_paths), start=1):
        doc_id = f"policy_{idx:03d}"
        manifest[doc_id] = {
            "document_name": path.stem,
            "source_path": str(path.relative_to(config.ROOT)),
            "department": "HR",
            "country": "Global",
            "version": "v1",
        }
    return manifest


def main():
    config.PROCESSED_TEXT.mkdir(parents=True, exist_ok=True)

    pdf_paths = list(config.RAW_POLICIES.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.RAW_POLICIES}")
        return

    manifest = build_manifest(pdf_paths)
    config.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest with {len(manifest)} documents -> {config.MANIFEST_FILE}")

    for doc_id, meta in manifest.items():
        pages = extract_pdf(config.ROOT / meta["source_path"])
        out_file = config.PROCESSED_TEXT / f"{doc_id}.json"
        out_file.write_text(json.dumps({"document_id": doc_id, "pages": pages}, indent=2))
        print(f"{doc_id}: {len(pages)} pages -> {out_file.name}")


if __name__ == "__main__":
    main()
