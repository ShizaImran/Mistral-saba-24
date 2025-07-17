# faiss_setup.py
import json
import os
import faiss
import pickle
import re
import argparse
from sentence_transformers import SentenceTransformer

# --- Configuration ---
RESOURCE_PATH = "utils/cleaned_study_resources(2).json"
INDEX_SAVE_PATH = "faiss_index/study_resources.index"
METADATA_SAVE_PATH = "faiss_index/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Clean Text Utility ---
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()

# --- FAISS Setup Logic ---
def create_faiss_index():
    if not os.path.exists(RESOURCE_PATH):
        print(f"❌ Resource file not found: {RESOURCE_PATH}")
        return

    print("📦 Loading resources...")
    with open(RESOURCE_PATH, "r", encoding="utf-8") as f:
        resources = json.load(f)

    print("🔎 Initializing embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    metadata = []

    print("🧠 Preparing text for embedding...")
    for item in resources:
        text = " ".join([
            clean_text(item.get("title", "")),
            clean_text(item.get("description", "")),
            clean_text(item.get("keywords", "")),
            clean_text(item.get("topic", "")),
            clean_text(item.get("subtopics", "")),
            clean_text(item.get("purpose", "")),
            clean_text(item.get("format", ""))
        ])
        texts.append(text)
        metadata.append(item)

    print("⚙️ Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("📂 Creating FAISS index...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("faiss_index", exist_ok=True)

    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(METADATA_SAVE_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ FAISS index and metadata saved successfully.")

# --- CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or refresh the FAISS study resource index.")
    parser.add_argument("--refresh", action="store_true", help="Rebuild the FAISS index and metadata.")
    args = parser.parse_args()

    if args.refresh:
        create_faiss_index()
    else:
        print("ℹ️ Use `--refresh` to build or update the FAISS index:")
        print("   python faiss_setup.py --refresh")
