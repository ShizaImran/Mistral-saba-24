# faiss_search.py
import faiss
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
INDEX_PATH = "faiss_index/study_resources.index"
METADATA_PATH = "faiss_index/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Load FAISS index and metadata ---
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# --- Load embedding model ---
model = SentenceTransformer(EMBEDDING_MODEL)

# --- Difficulty and Format Mappings ---
LEVEL_MAPPING = {
    "matric": ["basic", "beginner"],
    "intermediate": ["intermediate"],
    "undergraduate": ["advanced", "intermediate"]
}

STYLE_MAPPING = {
    "visual": ["video", "diagram", "pdf"],
    "auditory": ["audio", "lecture"],
    "reading/writing": ["text", "pdf", "article"],
    "kinesthetic": ["interactive", "lab"]
}

# --- Normalize and Tokenize Format ---
def normalize_format(fmt):
    if not fmt:
        return ""
    tokens = re.split(r"[\/\-]", fmt.lower())
    return " ".join(tokens)

# --- Filter Based on Student Profile ---
def is_relevant_to_student(resource, level, style):
    difficulty = (resource.get("difficulty") or "").lower()
    raw_format = (resource.get("format") or "")
    fmt_clean = normalize_format(raw_format)

    level_ok = any(tag in difficulty for tag in LEVEL_MAPPING.get(level, [])) or not difficulty
    style_ok = any(tag in fmt_clean for tag in STYLE_MAPPING.get(style, [])) or not fmt_clean

    return level_ok and style_ok

# --- Topic Matching ---
def matches_topic(resource, topic):
    topic = topic.lower()
    topic_fields = (resource.get("topic") or "") + " " + (resource.get("subtopics") or "")
    return topic in topic_fields.lower()

# --- Combined FAISS + Filtering Search ---
def search_resources(query, level="matric", style="visual", selected_topic=None, top_k=5):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype("float32"), top_k * 3)

    raw_results = [metadata[i] for i in I[0] if i < len(metadata)]

    filtered = [
        r for r in raw_results
        if is_relevant_to_student(r, level.lower(), style.lower()) and
           (selected_topic is None or matches_topic(r, selected_topic))
    ]

    return filtered[:top_k] if filtered else raw_results[:top_k]  # fallback

# --- Optional CLI Test ---
if __name__ == "__main__":
    query = "binary number video"
    results = search_resources(query, level="matric", style="visual", selected_topic="Binary")

    print("🔍 Top Recommendations:")
    for res in results:
        print(f"🔹 {res.get('title')} ({res.get('format', 'N/A')} | {res.get('difficulty', 'N/A')})")
