import streamlit as st
from firebase_init import db
from datetime import datetime

# Profile Page for Students
def profile_page():
    email = st.session_state.get("email")
    doc = db.collection("students").document(email).get()

    # Skip if already done
    if doc.exists and doc.to_dict().get("quiz_done"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.title("🧠 Complete Your Profile")

    name = st.text_input("Your Name")
    class_level = st.selectbox("Education Level", ["Matric", "Intermediate", "Undergraduate"])
    learning_style = st.selectbox("Prefered Contetn Type", ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
    # 📘 Book-aligned Domain and Topic Structure
    domain_topic_map = {
        "Binary & Number Systems": [
            "Introduction to Binary",
            "Denary to Binary Conversion",
            "Binary to Denary Conversion",
            "Hexadecimal Basics",
            "Binary vs Hexadecimal"
        ],
        "Communication & Internet": [
            "Serial & Parallel Transmission",
            "USB & Protocols",
            "HTML, HTTP, Web Browsers",
            "Error Checking Methods"
        ],
        "Logic & Circuits": [
            "Basic Logic Gates",
            "Truth Tables",
            "Logic Circuits in Real World",
            "XOR, NAND, NOR Applications"
        ],
        "Operating Systems": [
            "Functions of OS",
            "Interrupts & Buffers",
            "Fetch-Execute Cycle"
        ],
        "Input/Output Devices": [
            "Scanners & Cameras",
            "Printers & Projectors",
            "Sensors & Microphones",
            "Actuators & Touch Screens"
        ],
        "Storage & Memory": [
            "File Formats (JPEG, MP3, etc.)",
            "Lossless vs Lossy Compression",
            "Primary vs Secondary Storage"
        ],
        "Programming Languages": [
            "High-Level vs Low-Level",
            "Compilers vs Interpreters",
            "Syntax & Logic Errors"
        ],
        "Security & Ethics": [
            "Viruses & Hacking",
            "Encryption & Firewalls",
            "Computer Ethics & Privacy"
        ]
    }
    # ✅ Streamlit Inputs
    domain = st.selectbox("🎓 Domain of Interest", list(domain_topic_map.keys()))
    topic = st.selectbox("🔍 Specific Topic", domain_topic_map[domain])


    if st.button("Submit Profile"):
        if not name or not topic:
            st.error("All fields are required.")
            return

        db.collection("students").document(email).update({
            "name": name,
            "class_level": class_level,
            "learning_style": learning_style,
            "domain": domain,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        })

        st.success("Profile saved. Starting your quiz...")
        st.session_state.profile = {
            "name": name,
            "email": email,
            "class_level": class_level,
            "learning_style": learning_style,
            "domain": domain,
            "topic": topic
        }
        st.session_state.page = "quiz"
        st.rerun()
 # type: ignore