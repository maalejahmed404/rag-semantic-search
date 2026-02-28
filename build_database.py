"""
build_database.py
Ingests the extracted chunks JSON into a persistent ChromaDB collection.
Uses all-MiniLM-L6-v2 for embedding generation (384 dimensions).

KEY FIX: Each chunk is EXPLODED so that every ingredient gets its own entry.
This eliminates the multi-ingredient problem where filtering fails because
the ingredient metadata is a comma-separated list.

Before: 1 chunk with ingredients=["A", "B", "C"] -> 1 DB entry with ingredients="A, B, C"
After:  1 chunk with ingredients=["A", "B", "C"] -> 3 DB entries, each with ingredient="A", "B", "C"
"""
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(SCRIPT_DIR, "all_extracted_chunks_merged.json")
DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "ingredients_rag"


def chunk_to_document(chunk, single_ingredient):
    """Convert a JSON chunk into a rich, self-contained sentence for embedding.
    
    Uses a SINGLE ingredient name (not the full list) to create a precise
    semantic unit that can be correctly filtered and retrieved.
    """
    section = chunk.get("section title", "")

    if "table type" in chunk:
        row_title = chunk.get("row title", "")
        row_text = chunk.get("row text", "")
        return f"{single_ingredient}. {section}: {row_title} - {row_text}"
    else:
        text = chunk.get("chunk text", "")
        text = text.replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return f"{single_ingredient}. {text}"


def chunk_to_metadata(chunk, single_ingredient):
    """Extract metadata for ChromaDB with a SINGLE ingredient."""
    meta = {
        "update_date": chunk.get("update date", ""),
        "section_title": chunk.get("section title", ""),
        "ingredient": single_ingredient,  # SINGLE ingredient, not a list
    }
    if "table type" in chunk:
        meta["table_type"] = chunk.get("table type", "")
        meta["row_title"] = chunk.get("row title", "")
    return meta


def build():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} raw chunks from JSON.")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = []
    metadatas = []
    ids = []
    
    # EXPLODE: one entry per ingredient
    entry_id = 0
    multi_count = 0
    for i, chunk in enumerate(chunks):
        ingredients = chunk.get("ingredients", [])
        if not ingredients:
            ingredients = ["unknown"]
        
        if len(ingredients) > 1:
            multi_count += 1
        
        for ing in ingredients:
            doc_text = chunk_to_document(chunk, ing)
            if not doc_text.strip() or doc_text.strip() == ".":
                continue
            documents.append(doc_text)
            metadatas.append(chunk_to_metadata(chunk, ing))
            ids.append(f"chunk_{i}_ing_{entry_id}")
            entry_id += 1

    print(f"\nExplosion stats:")
    print(f"  Raw chunks: {len(chunks)}")
    print(f"  Multi-ingredient chunks: {multi_count}")
    print(f"  Total DB entries after explosion: {len(documents)}")

    # Show a few examples
    print(f"\nSample documents being embedded:")
    for d in documents[:5]:
        print(f"  -> {d[:120]}...")
    print()

    # Show sample metadata
    print(f"Sample metadata:")
    for m in metadatas[:5]:
        print(f"  -> ingredient='{m['ingredient']}', section='{m['section_title']}'")
    print()

    print(f"Embedding {len(documents)} documents ...")
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)

    batch_size = 5000
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist()
        )
    
    print(f"\nChromaDB collection '{COLLECTION_NAME}' built with {collection.count()} entries.")
    print(f"Persistent database saved to: {DB_PATH}")
    
    # Verify: check unique ingredients
    all_metas = collection.get(limit=collection.count(), include=["metadatas"])
    unique_ings = sorted(set(m["ingredient"] for m in all_metas["metadatas"]))
    print(f"\nUnique ingredients in DB: {len(unique_ings)}")
    for ing in unique_ings:
        count = sum(1 for m in all_metas["metadatas"] if m["ingredient"] == ing)
        print(f"  {ing}: {count} entries")


if __name__ == "__main__":
    build()
