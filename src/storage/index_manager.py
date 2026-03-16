"""
Step 3: The "Library" (Index Factory)
Now that you have chunks, you need to store them. Since you want multiple index types (Vector, Summary), you need a factory.

File: src/storage/index_manager.py

Class: IndexFactory

Methods:

create_vector_index(nodes): For quick fact-finding.

create_summary_index(nodes): For "What is this whole chapter about?"

get_engine(index_type): Returns the correct QueryEngine based on the user's intent.

Why: This follows the Open/Closed Principle. You can add a KnowledgeGraphIndex later without changing your ingestion code.
"""
# The "Human-in-the-Loop" logic might be implemented soon
import asyncio
import shutil
import chromadb
from pathlib import Path
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

class IndexFactory:
    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.storage_dir / "chroma_db")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
    def _get_vector_store(self, subject: str):
        subject = subject.lower()
        chroma_collection = self.client.get_or_create_collection(f"{subject}_collection")
        return ChromaVectorStore(chroma_collection=chroma_collection)

    def get_or_create_index(self, nodes=None, index_type="vector", subject="general"):
        subject = subject.lower()
        collection_name = f"{subject}_collection"
        
        if index_type == "vector":
            # ✅ Bug 4 fix: explicit None check
            if nodes is not None:
                vector_store = self._get_vector_store(subject)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                print(f"📦 Indexing {len(nodes)} nodes into {collection_name}...")
                return VectorStoreIndex(nodes, storage_context=storage_context)

            collections = [c.name for c in self.client.list_collections()]
            print(f"DEBUG collections: {collections}")

            if collection_name in collections:
                vector_store = self._get_vector_store(subject)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                col = self.client.get_collection(collection_name)
                count = col.count()

                if count > 0:
                    print(f"📖 Found {count} points in {collection_name}. Loading index...")
                    return VectorStoreIndex.from_vector_store(
                        vector_store,
                        storage_context=storage_context
                    )

            # ✅ Bug 3 fix: raise instead of returning None
            raise ValueError(
                f"No 'vector' index found for subject '{subject}'. "
                f"Run ingestion first or check storage path: {self.db_path}"
            )

        
    def get_full_text(self, subject: str) -> str:
        path = self.storage_dir / "full_text" / f"{subject.lower()}.txt"
        if not path.exists():
            raise ValueError(
                f"No full text found for subject: '{subject}'. "
                f"Make sure the PDF was uploaded and processed first."
            )
        return path.read_text(encoding="utf-8")

    async def get_query_engine(self, subject: str, index_type: str = "vector", streaming: bool = False):
        subject = subject.lower()
        index = self.get_or_create_index(nodes=None, index_type=index_type, subject=subject)

        # ✅ Bug 1 fix: MetadataFilter instead of ExactMatchFilter
        filters = MetadataFilters(filters=[
            MetadataFilter(key="subject", value=subject)
        ])

        return index.as_query_engine(
            filters=filters,
            streaming=streaming,
            similarity_top_k=3,
            response_mode="compact"
        )
    

    # async def delete_document(self, subject: str, file_name: str):
    #     """Removes a specific file from Chroma via metadata."""
    #     # We access the collection directly through the client to delete by metadata
    #     collection = self.client.get_collection(f"{subject}_collection")
        
    #     # Chroma deletion via metadata filter
    #     collection.delete(where={"file_name": file_name})
    #     print(f"🗑️ Deleted {file_name} from {subject} (Chroma).")

    # async def update_document(self, nodes, subject: str, file_name: str):
    #     """Deletes and then re-indexes a document."""
    #     await self.delete_document(subject, file_name)
    #     self.get_or_create_index(nodes, index_type="vector", subject=subject)
    #     print(f"🔄 Updated {file_name} in {subject}.")

    # async def clear_subject(self, subject: str):
    #     """Wipes a specific collection from Chroma."""
    #     try:
    #         self.client.delete_collection(f"{subject}_collection")
    #         print(f"🔥 Subject '{subject}' removed from Chroma.")
    #     except Exception as e:
    #         print(f"⚠️ Could not delete collection: {e}")

    # async def clear_all(self):
    #     """Wipes the entire database directory."""
    #     if self.storage_dir.exists():
    #         # Chroma doesn't lock files as aggressively as Qdrant,
    #         # but we still use thread-safe deletion.
    #         await asyncio.to_thread(shutil.rmtree, self.storage_dir)
    #         self.storage_dir.mkdir(parents=True, exist_ok=True)
            
    #         # Reset the client
    #         self.client = chromadb.PersistentClient(path=self.db_path)
    #         print("🚀 Chroma Database wiped clean!")
# -----------------------------------------------------


import asyncio
import json
import os
from pathlib import Path
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from src.storage.index_manager import IndexFactory

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
TEST_STORAGE_PATH = "./test_storage_chroma"
SUBJECT           = "nlp"
NODES_PATH        = "nodes.json"
FULL_TEXT_PATH    = "./storage/full_text/nlp.txt"


async def test_index_factory():

    print("=" * 55)
    print("   INDEX FACTORY — PIPELINE TEST (Use Case B)")
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
        results_log.append(f"  {icon}  {check:<50} {detail}")

    # ─────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    factory = IndexFactory(storage_dir=TEST_STORAGE_PATH)

    # ─────────────────────────────────────────
    # STEP 1 — Load nodes from JSON
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 1 — Loading nodes from '{NODES_PATH}'...")
    nodes = []
    try:
        if not Path(NODES_PATH).exists():
            raise FileNotFoundError(f"{NODES_PATH} not found. Run the parser test first.")

        with open(NODES_PATH, "r") as f:
            node_dicts = json.load(f)

        nodes = []
        for nd in node_dicts:
            node = TextNode.from_dict(nd)
            node.metadata["subject"] = SUBJECT
            nodes.append(node)

        log("Nodes loaded from JSON", "PASS", f"{len(nodes)} nodes loaded")
    except Exception as e:
        log("Nodes loaded from JSON", "FAIL", f"Exception: {e}")
        print(f"   💥 Cannot continue without nodes: {e}")
        _print_report(results_log, passed, failed)
        return

    # ─────────────────────────────────────────
    # STEP 2 — Index nodes into ChromaDB
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 2 — Indexing {len(nodes)} nodes into ChromaDB...")
    try:
        index = factory.get_or_create_index(nodes=nodes, index_type="vector", subject=SUBJECT)
        if index is not None:
            log("Indexing into ChromaDB", "PASS", f"Collection: {SUBJECT}_collection")
        else:
            log("Indexing into ChromaDB", "FAIL", "get_or_create_index returned None")
    except Exception as e:
        log("Indexing into ChromaDB", "FAIL", f"Exception: {e}")
        print(f"   💥 Indexing failed — cannot continue: {e}")
        _print_report(results_log, passed, failed)
        return

    # ─────────────────────────────────────────
    # STEP 3 — Reload index from ChromaDB (no nodes)
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 3 — Reloading index from ChromaDB (nodes=None)...")
    try:
        loaded_index = factory.get_or_create_index(nodes=None, index_type="vector", subject=SUBJECT)
        if loaded_index is not None:
            log("Reload index from ChromaDB", "PASS", "Index loaded without passing nodes")
        else:
            log("Reload index from ChromaDB", "FAIL", "Returned None on reload")
    except Exception as e:
        log("Reload index from ChromaDB", "FAIL", f"Exception: {e}")

    # ─────────────────────────────────────────
    # STEP 4 — get_query_engine returns usable engine
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 4 — Testing get_query_engine()...")
    try:
        engine = await factory.get_query_engine(subject=SUBJECT, index_type="vector")
        if engine is not None:
            log("get_query_engine() returns engine", "PASS", "Query engine created successfully")
        else:
            log("get_query_engine() returns engine", "FAIL", "Returned None")
    except Exception as e:
        log("get_query_engine() returns engine", "FAIL", f"Exception: {e}")

    # ─────────────────────────────────────────
    # STEP 5 — Query engine returns results
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 5 — Running a test query through the engine...")
    try:
        engine = await factory.get_query_engine(subject=SUBJECT, index_type="vector")
        response = await engine.aquery("What is NLP?")
        answer = str(response).strip()
        if len(answer) > 0:
            log("Query engine returns results", "PASS", f"Response: {answer[:80]}...")
        else:
            log("Query engine returns results", "FAIL", "Empty response")
    except Exception as e:
        log("Query engine returns results", "FAIL", f"Exception: {e}")

    # ─────────────────────────────────────────
    # STEP 6 — get_full_text returns content
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 6 — Testing get_full_text() for subject '{SUBJECT}'...")
    try:
        # Copy full text to test storage so factory can find it
        src = Path(FULL_TEXT_PATH)
        dst = Path(TEST_STORAGE_PATH) / "full_text" / f"{SUBJECT}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.exists() and not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        full_text = factory.get_full_text(SUBJECT)
        word_count = len(full_text.split())
        if len(full_text) > 0:
            log("get_full_text() returns content", "PASS", f"{len(full_text)} chars, ~{word_count} words")
        else:
            log("get_full_text() returns content", "FAIL", "Returned empty string")
    except Exception as e:
        log("get_full_text() returns content", "FAIL", f"Exception: {e}")

    # ─────────────────────────────────────────
    # STEP 7 — get_full_text raises on missing subject
    # ─────────────────────────────────────────
    print(f"\n▸ STEP 7 — Testing get_full_text() raises on unknown subject...")
    try:
        factory.get_full_text("subject_that_does_not_exist")
        log("get_full_text() raises on missing subject", "FAIL", "No exception raised — should have raised ValueError")
    except ValueError as e:
        log("get_full_text() raises on missing subject", "PASS", f"ValueError raised correctly")
    except Exception as e:
        log("get_full_text() raises on missing subject", "FAIL", f"Wrong exception type: {e}")

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
    asyncio.run(test_index_factory())