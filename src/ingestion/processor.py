"""
Step 2: The "Ingestion Engine" (Parser)
You cannot have a good RAG without good data. This is where you handle the "Multi-Modal" aspect.

File: src/ingestion/processor.py

Class: DocumentProcessor

Methods:

__init__(api_key): Initialize LlamaParse.

load_and_parse(file_path): Converts PDF to Markdown.

chunk_markdown(content): Break the Markdown into logical sections (chapters/headers) rather than just character counts.

Why: Using Markdown headers as "boundaries" for chunks is much more "industry-standard" than random 1000-character splits.
"""
from pathlib import Path
import time
import os
from datetime import datetime, timezone
from typing import List
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
from src.schemas.models import StudyMaterialMetadata
import asyncio
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

# Fields injected into LLM prompt (for citations and context)
LLM_METADATA_KEYS = ["file_name", "subject", "page_number"]

# Fields injected into embedding (only semantic content fields)
EMBED_METADATA_KEYS = ["subject"]


class DocumentProcessor:
    def __init__(self, api_key: str = None):
        self.parser = LlamaParse(
            api_key=api_key or os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            split_by_page=True,
            tier="agentic",
            version="latest",

            processing_options={
                "specialized_chart_parsing": "agentic",
                "ocr_parameters": {"languages": ["en"]},
                "ignore": {"ignore_diagonal_text": True}
            },

            output_options={
                "markdown": {
                    "tables": {
                        "output_tables_as_markdown": True,
                        "merge_continued_tables": True
                    }
                },
                "images_to_save": ["embedded"],
            },

            agentic_options={
                "custom_prompt": """
                    You are a university study mentor assistant.
                    - Convert all math to LaTeX using double dollar signs $$ for blocks.
                    - Transform flowcharts or process diagrams into Mermaid.js code blocks.
                    - If a page has a footer or header with the course name, ignore it.
                    - Provide a detailed [ALT TEXT] for any complex diagrams that cannot be Mermaid-fied.
                """
            }
        )


    async def load_and_parse(self, file_path: str, subject: str) -> List:
        documents = await self.parser.aload_data(file_path)

        file_name      = os.path.basename(file_path)
        file_stats     = os.stat(file_path)
        file_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_stats.st_mtime))
        upload_date    = datetime.now(timezone.utc).isoformat()

        for i, doc in enumerate(documents):
            raw_page = doc.metadata.get("page_number") or doc.metadata.get("page_label")

            if raw_page is None:
                page_int = i + 1
            else:
                try:
                    page_int = int(raw_page)
                except ValueError:
                    page_int = i + 1

            meta_obj = StudyMaterialMetadata(
                file_id=doc.doc_id,
                file_name=file_name,
                subject=subject,
                page_number=page_int,
                timestamp=file_timestamp,
                upload_date=upload_date,
            )

            doc.metadata = meta_obj.model_dump(mode="json")

        return documents

    async def process(self, file_path: str, subject: str) -> List:
        """Full pipeline: parse PDF → apply metadata → save full text → return page nodes."""
        documents = await self.load_and_parse(file_path, subject)

        # ✅ Save full text from raw pages (before any chunking)
        full_text = "\n\n".join([doc.get_content() for doc in documents])
        self.save_full_text(subject, full_text)

        # ✅ Convert documents directly to TextNodes (one per page, no chunking)
        nodes = []
        for doc in documents:
            node = TextNode(
                text=doc.get_content(),
                metadata=doc.metadata,
                id_=doc.doc_id
            )
            node.excluded_llm_metadata_keys = [
                k for k in doc.metadata.keys() if k not in LLM_METADATA_KEYS
            ]
            node.excluded_embed_metadata_keys = [
                k for k in doc.metadata.keys() if k not in EMBED_METADATA_KEYS
            ]
            nodes.append(node)

        print(f"✅ Processed '{os.path.basename(file_path)}' → {len(nodes)} page nodes")
        return nodes

    def save_full_text(self, subject: str, text: str):
        path = Path(f"./storage/full_text/{subject}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"💾 Full text saved for subject: '{subject}'")
# ----------------------------------------------------------------------
import json
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FILE_PATH     = "data/2-ML-FOR-NLP-2022.pdf"
SUBJECT       = "nlp"
SAVE_PATH     = "nodes.json"
FULL_TEXT_PATH = f"./storage/full_text/{SUBJECT}.txt"
REQUIRED_KEYS = ["file_id", "file_name", "subject", "timestamp", "page_number", "upload_date"]


async def test_document_processor():

    print("=" * 55)
    print("   DOCUMENT PROCESSOR — PIPELINE TEST (Use Case B)")
    print("=" * 55)

    passed      = 0
    failed      = 0
    results_log = []

    def log(check: str, status: str, detail: str):
        nonlocal passed, failed
        icon = "✅" if status == "PASS" else "❌"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        results_log.append(f"  {icon}  {check:<45} {detail}")

    processor = DocumentProcessor(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

    # ─────────────────────────────────────────
    # STEP 1 — Run full process() pipeline
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 1 — Running process() on '{FILE_PATH}'...")
    nodes = []
    try:
        nodes = await processor.process(FILE_PATH, SUBJECT)

        if len(nodes) > 0:
            log("process() returned nodes", "PASS", f"{len(nodes)} page nodes created")
        else:
            log("process() returned nodes", "FAIL", "process() returned 0 nodes")

    except Exception as e:
        log("process() returned nodes", "FAIL", f"Exception: {e}")
        print(f"   💥 process() failed — cannot continue: {e}")
        _print_report(results_log, passed, failed)
        return

    # ─────────────────────────────────────────
    # STEP 2 — One node per page check
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 2 — Verifying one node per page...")
    page_numbers = [n.metadata.get("page_number") for n in nodes]
    unique_pages = set(page_numbers)

    if len(nodes) == len(unique_pages):
        log("One node per page (no duplicates)", "PASS", f"{len(nodes)} nodes, {len(unique_pages)} unique pages")
    else:
        log("One node per page (no duplicates)", "FAIL", f"{len(nodes)} nodes but {len(unique_pages)} unique pages")

    # ─────────────────────────────────────────
    # STEP 3 — Validate Node Metadata
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 3 — Validating metadata on sample node...")
    sample_node = nodes[0]
    sample_meta = sample_node.metadata
    print(f"   Metadata: {sample_meta}")

    for key in REQUIRED_KEYS:
        val = sample_meta.get(key)
        if val is not None and val != "":
            log(f"Metadata key present: '{key}'", "PASS", str(val))
        else:
            log(f"Metadata key present: '{key}'", "FAIL", "Missing or empty")

    # ─────────────────────────────────────────
    # STEP 4 — Validate Metadata Filter Keys
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 4 — Validating LLM & Embed metadata filter keys...")

    llm_excluded   = sample_node.excluded_llm_metadata_keys
    embed_excluded = sample_node.excluded_embed_metadata_keys

    if "subject" not in embed_excluded:
        log("'subject' included in embeddings", "PASS", "Correct — used for filtering")
    else:
        log("'subject' included in embeddings", "FAIL", "Should not be in excluded_embed_metadata_keys")

    if "file_name" not in llm_excluded:
        log("'file_name' visible to LLM", "PASS", "Correct — used for citations")
    else:
        log("'file_name' visible to LLM", "FAIL", "Should not be in excluded_llm_metadata_keys")

    if "page_number" not in llm_excluded:
        log("'page_number' visible to LLM", "PASS", "Correct — used for citations")
    else:
        log("'page_number' visible to LLM", "FAIL", "Should not be in excluded_llm_metadata_keys")

    # ─────────────────────────────────────────
    # STEP 5 — Validate Subject Consistency
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 5 — Checking subject consistency across all nodes...")
    wrong_subject = [
        i for i, n in enumerate(nodes)
        if n.metadata.get("subject", "").lower() != SUBJECT.lower()
    ]
    if not wrong_subject:
        log("Subject consistent across all nodes", "PASS", f"All {len(nodes)} nodes have subject='{SUBJECT}'")
    else:
        log("Subject consistent across all nodes", "FAIL", f"{len(wrong_subject)} nodes have wrong subject")

    # ─────────────────────────────────────────
    # STEP 6 — Validate Page Number Coverage
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 6 — Checking page number coverage...")
    missing_pages = [i for i, n in enumerate(nodes) if n.metadata.get("page_number") is None]
    if not missing_pages:
        sorted_pages = sorted(unique_pages)
        log("All nodes have page numbers", "PASS", f"Pages covered: {sorted_pages[0]}–{sorted_pages[-1]}")
    else:
        log("All nodes have page numbers", "FAIL", f"{len(missing_pages)} nodes missing page_number")

    # ─────────────────────────────────────────
    # STEP 7 — Verify full text file was saved
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 7 — Verifying full text file saved to '{FULL_TEXT_PATH}'...")
    full_text_path = Path(FULL_TEXT_PATH)

    if full_text_path.exists():
        content = full_text_path.read_text(encoding="utf-8")
        word_count = len(content.split())
        if len(content) > 0:
            log("Full text file created", "PASS", f"{len(content)} chars, ~{word_count} words")
        else:
            log("Full text file created", "FAIL", "File exists but is empty")
    else:
        log("Full text file created", "FAIL", f"File not found at {FULL_TEXT_PATH}")

    # ─────────────────────────────────────────
    # STEP 8 — Full text covers all pages
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 8 — Verifying full text contains content from all pages...")
    if full_text_path.exists():
        full_text = full_text_path.read_text(encoding="utf-8")
        # Each page node text should appear in the full text
        missing_in_fulltext = [
            i for i, n in enumerate(nodes)
            if n.get_content()[:50] not in full_text
        ]
        if not missing_in_fulltext:
            log("Full text contains all page content", "PASS", f"All {len(nodes)} pages accounted for")
        else:
            log("Full text contains all page content", "FAIL", f"{len(missing_in_fulltext)} pages missing from full text")
    else:
        log("Full text contains all page content", "FAIL", "Full text file not found — skipping")

    # ─────────────────────────────────────────
    # STEP 9 — Save Nodes to JSON
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 9 — Saving nodes to '{SAVE_PATH}'...")
    try:
        with open(SAVE_PATH, "w") as f:
            json.dump([node.dict() for node in nodes], f, indent=2)
        log(f"Nodes saved to '{SAVE_PATH}'", "PASS", f"{len(nodes)} nodes written")
        print(f"   💾 Saved to {SAVE_PATH}")
    except Exception as e:
        log(f"Nodes saved to '{SAVE_PATH}'", "FAIL", f"Exception: {e}")

    # ─────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────
    _print_report(results_log, passed, failed)


def _print_report(results_log: list, passed: int, failed: int):
    total = passed + failed
    print("\n" + "=" * 55)
    print("   TEST REPORT")
    print("=" * 55)
    for line in results_log:
        print(line)
    print("-" * 55)
    print(f"   Total: {total} | Passed: {passed} ✅ | Failed: {failed} ❌")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(test_document_processor())