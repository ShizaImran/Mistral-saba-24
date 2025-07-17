# learning_plan.py
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# === Load Environment ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# === Configuration ===
INDEX_FILE = "faiss_index/study_resources.index"
METADATA_FILE = "faiss_index/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8
LLM_MODEL = "mistral-saba-24b"

# === Chapter-Topic Mapping ===
CHAPTER_TOPICS = {
    "Chapter 1: Binary Systems and Hexadecimal": [
        "Introduction to Binary",
        "Denary to Binary Conversion",
        "Binary to Denary Conversion",
        "Hexadecimal Basics",
        "Binary vs Hexadecimal"
    ],
    "Chapter 2: Communication and Internet Technologies": [
        "Serial & Parallel Transmission",
        "USB & Protocols",
        "HTML, HTTP, Web Browsers",
        "Error Checking Methods"
    ],
    "Chapter 3: Logic Gates and Logic Circuits": [
        "Basic Logic Gates",
        "Truth Tables",
        "Logic Circuits in Real World",
        "XOR, NAND, NOR Applications"
    ],
    "Chapter 4: Operating Systems and Computer Architecture": [
        "Functions of OS",
        "Interrupts & Buffers",
        "Fetch-Execute Cycle"
    ],
    "Chapter 5: Input and Output Devices": [
        "Scanners & Cameras",
        "Printers & Projectors",
        "Sensors & Microphones",
        "Actuators & Touch Screens"
    ],
    "Chapter 6: Memory and Data Storage": [
        "File Formats (JPEG, MP3, etc.)",
        "Lossless vs Lossy Compression",
        "Primary vs Secondary Storage"
    ],
    "Chapter 7: High- and Low-Level Languages": [
        "High-Level vs Low-Level",
        "Compilers vs Interpreters",
        "Syntax & Logic Errors"
    ],
    "Chapter 8: Security and Ethics": [
        "Viruses & Hacking",
        "Encryption & Firewalls",
        "Computer Ethics & Privacy"
    ]
}

# === Load Models and Data ===
@st.cache_resource
def load_models():
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_FILE)
    return model, index

@st.cache_data
def load_metadata():
    with open(METADATA_FILE, "rb") as f:
        return pickle.load(f)

model, index = load_models()
metadata = load_metadata()

# === Core Functions ===
def retrieve_resources(query: str, k=TOP_K):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

def generate_study_plan(style, chapter, topic, resources):
    try:
        resource_list = []
        for resource in resources[:10]:
            if isinstance(resource, (np.ndarray, np.generic)):
                resource = resource.item() if resource.size == 1 else resource.tolist()
            resource_list.append(resource)
        
        prompt = f"""
**Task:** Create a detailed study plan for Computer Science students.

**Chapter:** {chapter}
**Topic:** {topic}
**Learning Style:** {style}

**Available Resources:**
{resource_list}

**Instructions:**
1. Create a 5-step learning path
2. Include both theoretical and practical components
3. Suggest assessment methods
4. Format in clear markdown with emojis
5. Keep it concise (1-2 pages equivalent)
6. Focus specifically on: {topic}
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a computer science teaching expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating plan: {str(e)}")
        return None

# === Streamlit UI ===
def learning_plan_page():
    st.title("📚 Computer Science Learning Plan Generator")
    
    # Initialize session state for selections
    if 'selected_chapter' not in st.session_state:
        st.session_state.selected_chapter = list(CHAPTER_TOPICS.keys())[0]
    
    # Chapter selection
    selected_chapter = st.selectbox(
        "📘 Select Chapter",
        options=list(CHAPTER_TOPICS.keys()),
        index=0,
        key='chapter_select'
    )
    
    # Update topics when chapter changes
    if selected_chapter != st.session_state.selected_chapter:
        st.session_state.selected_chapter = selected_chapter
        st.session_state.selected_topic = CHAPTER_TOPICS[selected_chapter][0]
    
    # Topic selection
    selected_topic = st.selectbox(
        "🔍 Select Topic",
        options=CHAPTER_TOPICS[selected_chapter],
        index=0,
        key='topic_select'
    )
    
    # Learning style selection
    learning_style = st.selectbox(
        "🎨 Learning Style",
        options=["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
        index=0
    )
    
    if st.button("✨ Generate Learning Plan", type="primary"):
        with st.spinner("Creating your personalized learning plan..."):
            # Retrieve resources
            search_query = f"{selected_chapter}: {selected_topic}"
            resources = retrieve_resources(search_query)
            
            # Generate plan
            plan = generate_study_plan(learning_style, selected_chapter, selected_topic, resources)
            
            if plan:
                # Display results
                st.subheader(f"📖 {selected_chapter}: {selected_topic}")
                st.markdown("---")
                st.markdown(plan)
                
                # Show resources
                st.subheader("📚 Recommended Resources")
                for i, resource in enumerate(resources[:5], 1):
                    if isinstance(resource, dict):
                        title = resource.get('title', 'Untitled Resource')
                        url = resource.get('url', '#')
                        st.markdown(f"{i}. [{title}]({url})")
                        if 'description' in resource:
                            st.caption(resource['description'])
                    else:
                        st.markdown(f"{i}. {str(resource)}")
            else:
                st.error("Failed to generate learning plan. Please try again.")

if __name__ == "__main__":
    learning_plan_page()