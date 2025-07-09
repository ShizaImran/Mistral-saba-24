import streamlit as st
import json
import re
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from datetime import datetime
from firebase_init import db, get_progress_ref

# --- Configuration ---
INDEX_FILE = "faiss_index/study_resources.index"
METADATA_FILE = "faiss_index/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def load_recommendation_engine():
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

try:
    model, index, metadata = load_recommendation_engine()
except Exception as e:
    st.error(f"Failed to load recommendation engine: {e}")
    st.stop()

# --- Mappings ---
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

# --- Format Normalization ---
def normalize_format(fmt):
    if not fmt:
        return ""
    tokens = re.split(r"[\/\-]", fmt.lower())
    return " ".join(tokens)

# --- Student Preference Filter ---
def is_relevant_to_student(resource, level, style):
    difficulty = (resource.get("difficulty") or "").lower()
    raw_format = (resource.get("format") or "")
    fmt_clean = normalize_format(raw_format)

    level_ok = any(tag in difficulty for tag in LEVEL_MAPPING.get(level, [])) or not difficulty
    style_ok = any(tag in fmt_clean for tag in STYLE_MAPPING.get(style, [])) or not fmt_clean

    return level_ok and style_ok

# --- FAISS Recommendation Engine ---
def get_faiss_recommendations(query, preferred_style, level, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    raw = [metadata[i] for i in I[0] if i < len(metadata)]
    filtered = [r for r in raw if is_relevant_to_student(r, level, preferred_style)]
    return filtered if filtered else raw[:k]  # fallback to top-k unfiltered

# --- Log to Firebase ---
def log_recommendation(email, topic, resources):
    ref = db.collection("students").document(email).collection("recommendation_history")
    ref.add({
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "resources": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "format": r.get("format"),
                "difficulty": r.get("difficulty"),
                "duration": r.get("duration"),
                "topic_name": r.get("topic"),
            }
            for r in resources
        ],
        "viewed": False
    })

# --- Fetch History ---
def get_recommendation_history(email):
    ref = db.collection("students").document(email).collection("recommendation_history")
    return [dict(doc.to_dict(), id=doc.id) for doc in ref.order_by("timestamp", direction="DESCENDING").limit(10).stream()]

# --- Main Page ---
def recommendation_page():
    st.title("📚 Personalized Recommendations")

    if "email" not in st.session_state:
        st.error("Please log in first.")
        return

    email = st.session_state.email
    student_doc = db.collection("students").document(email).get()
    if not student_doc.exists:
        st.error("Student profile not found.")
        return

    student = student_doc.to_dict()
    preferred_style = student.get("learning_style", "visual").lower()
    level = student.get("class_level", "matric").lower()

    try:
        with open("utils/cleaned_study_resources(2).json", "r", encoding="utf-8") as f:
            content = json.load(f)
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return

    weak_topics = []
    for doc in get_progress_ref(email).collection("topics").stream():
        data = doc.to_dict()
        if data.get("status", "").lower() in ["weak", "needs improvement"]:
            weak_topics.append({"id": data.get("topic_id", ""), "name": doc.id})

    if not weak_topics:
        st.success("🎉 No weak topics found!")
        return

    with st.expander("📜 Recommendation History"):
        history = get_recommendation_history(email)
        if history:
            for entry in history:
                dt = datetime.fromisoformat(entry["timestamp"])
                st.markdown(f"**{dt.strftime('%Y-%m-%d %H:%M')}** - {entry['topic']}")
                for res in entry["resources"]:
                    st.markdown(f"- [{res['title']}]({res['url']})")
                st.markdown("---")
        else:
            st.info("No recommendation history yet")

    st.subheader("🔍 New Recommendations")
    selected_topic = st.selectbox("Select topic:", options=[t["name"] for t in weak_topics])

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Finding best resources..."):
            results = get_faiss_recommendations(
                query=f"{selected_topic} {preferred_style}",
                preferred_style=preferred_style,
                level=level
            )

            if results:
                log_recommendation(email, selected_topic, results)
                st.subheader(f"📚 Suggested Resources for: {selected_topic}")
                for r in results:
                    duration = round(float(r.get("duration", 0)))
                    st.markdown(f"### 🔹 {r.get('title', 'Untitled')}")
                    st.markdown(f"📄 *{r.get('description', '').strip()}*")
                    st.markdown(
                        f"🎯 **Format:** {r.get('format', 'N/A')} | "
                        f"🧠 **Difficulty:** {r.get('difficulty', 'N/A')} | "
                        f"⏱ **Duration:** {duration} min"
                    )
                    st.markdown(f"[🔗 Open Resource]({r.get('url', '#')})")
                    st.markdown("---")
            else:
                st.warning("No personalized matches found, showing general resources.")

# Run the page
if __name__ == "__main__":
    recommendation_page()
