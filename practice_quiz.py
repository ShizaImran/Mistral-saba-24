import streamlit as st
from firebase_init import db
from groq_api import generate_practice_quiz
from evaluate import evaluate_practice_quiz, update_subtopic_progress
from datetime import datetime, date, timedelta
import json
from recommender import get_faiss_recommendations

# --- Helpers for CLOs and PLOs ---
def get_clo_description(clo_id):
    try:
        with open("utils/clos.json", "r", encoding="utf-8") as f:
            clos = json.load(f)
        clo = next((c for c in clos if c["id"] == clo_id), None)
        return clo["description"] if clo else "N/A"
    except:
        return "N/A"

def get_plo_description(plo_id):
    try:
        with open("utils/plos.json", "r", encoding="utf-8") as f:
            plos = json.load(f)
        plo = next((p for p in plos if p["id"] == plo_id), None)
        return plo["description"] if plo else "N/A"
    except:
        return "N/A"

# --- Gamification Display ---
def display_gamification_profile(profile):
    st.sidebar.markdown(f"### 🎮 Your Learning Profile")
    st.sidebar.markdown(f"⭐ **Points:** {profile.get('points', 0)}")
    if profile.get("streak", 0) > 0:
        st.sidebar.markdown(f"🔥 **Streak:** {profile['streak']} days")
    if badges := profile.get("badges", []):
        st.sidebar.markdown("### 🏅 Your Badges")
        try:
            with open("utils/badges.json", "r", encoding="utf-8") as f:
                badge_defs = json.load(f)
            for badge_id in badges:
                badge = next((b for b in badge_defs if b["id"] == badge_id), None)
                if badge:
                    st.sidebar.markdown(f"{badge['icon']} **{badge['name']}**")
        except:
            st.sidebar.markdown("(Badges unavailable)")

# --- Badge Logic ---
def award_badges(email, profile, percent, topic, earned_badges):
    new_badges = []
    try:
        with open("utils/badges.json", "r", encoding="utf-8") as f:
            badge_defs = json.load(f)
    except:
        badge_defs = []

    if "first_quiz" not in earned_badges:
        new_badges.append("first_quiz")
    if "binary" in topic.lower() and percent >= 80 and "binary_pro" not in earned_badges:
        new_badges.append("binary_pro")
    if percent >= 90:
        qm_count = profile.get("quiz_master_count", 0) + 1
        db.collection("students").document(email).update({"quiz_master_count": qm_count})
        if qm_count >= 3 and "quiz_master" not in earned_badges:
            new_badges.append("quiz_master")
    streak = profile.get("streak", 0)
    if streak >= 3 and "streak_3" not in earned_badges:
        new_badges.append("streak_3")
    elif streak >= 7 and "streak_7" not in earned_badges:
        new_badges.append("streak_7")

    return new_badges, list(set(earned_badges + new_badges))

# --- Main Quiz Page ---
def practice_quiz_page():
    st.title("🧪 Practice Quiz")
    email = st.session_state.get("email")
    if not email:
        st.error("Login session expired. Please log in again.")
        return

    doc = db.collection("students").document(email).get()
    if not doc.exists:
        st.error("Student profile not found.")
        return

    profile = doc.to_dict()
    st.session_state.profile = profile
    display_gamification_profile(profile)

    topics_map = {
        "Chapter 1: Binary Systems and Hexadecimal": ["Introduction to Binary", "Denary to Binary Conversion", "Binary to Denary Conversion", "Hexadecimal Basics", "Binary vs Hexadecimal"],
        "Chapter 2: Communication and Internet Technologies": ["Serial & Parallel Transmission", "USB & Protocols", "HTML, HTTP, Web Browsers", "Error Checking Methods"],
        "Chapter 3: Logic Gates and Logic Circuits": ["Basic Logic Gates", "Truth Tables", "Logic Circuits in Real World", "XOR, NAND, NOR Applications"],
        "Chapter 4: Operating Systems and Computer Architecture": ["Functions of OS", "Interrupts & Buffers", "Fetch-Execute Cycle"],
        "Chapter 5: Input and Output Devices": ["Scanners & Cameras", "Printers & Projectors", "Sensors & Microphones", "Actuators & Touch Screens"],
        "Chapter 6: Memory and Data Storage": ["File Formats (JPEG, MP3, etc.)", "Lossless vs Lossy Compression", "Primary vs Secondary Storage"],
        "Chapter 7: High- and Low-Level Languages": ["High-Level vs Low-Level", "Compilers vs Interpreters", "Syntax & Logic Errors"],
        "Chapter 8: Security and Ethics": ["Viruses & Hacking", "Encryption & Firewalls", "Computer Ethics & Privacy"]
    }

    chapter = st.selectbox("📘 Choose Chapter", list(topics_map.keys()))
    topic = st.selectbox("🔍 Choose Topic", topics_map[chapter])

    if st.button("Generate Quiz"):
        questions = generate_practice_quiz(profile, chapter, topic)
        for q in questions:
            q["topic"] = topic
        st.session_state.practice_questions = questions
        st.session_state.practice_meta = {"chapter": chapter, "topic": topic}
        st.session_state.quiz_start_time = datetime.now()
        st.rerun()

    if "practice_questions" in st.session_state:
        st.subheader("🤩 Answer These Questions")
        questions = st.session_state.practice_questions
        answers = []

        for i, q in enumerate(questions):
            st.markdown(f"*Q{i+1}: {q.get('question', '')}*")
            options = q.get("options", [])
            selected = st.radio("Choose an option:", options, key=f"q{i}")
            index = options.index(selected) if selected in options else -1
            answers.append(chr(65 + index) if index != -1 else "-")

        if st.button("Submit Quiz"):
            duration = (datetime.now() - st.session_state.quiz_start_time).total_seconds()
            percent, weak_topics = evaluate_practice_quiz(questions, answers)
            update_subtopic_progress(email, percent, topic, chapter, duration)

            points = int(percent / 10)
            db.collection("students").document(email).update({"points": profile.get("points", 0) + points})
            st.success(f"🎉 You earned *+{points} points*!")

            today_str = date.today().isoformat()
            last_date = profile.get("last_practice_date", "")
            streak = profile.get("streak", 0)
            if last_date == (date.today() - timedelta(days=1)).isoformat():
                streak += 1
            elif last_date != today_str:
                streak = 1

            new_badges, earned_badges = award_badges(email, profile, percent, topic, profile.get("badges", []))

            db.collection("students").document(email).update({
                "badges": earned_badges,
                "streak": streak,
                "last_practice_date": today_str
            })

            st.success(f"✅ Your score: {percent}%")
            if new_badges:
                st.balloons()
                st.markdown("### 🏅 New Badges")
                try:
                    with open("utils/badges.json", "r", encoding="utf-8") as f:
                        badge_defs = json.load(f)
                    for b_id in new_badges:
                        b = next((b for b in badge_defs if b["id"] == b_id), None)
                        if b:
                            st.markdown(f"{b['icon']} *{b['name']}* – {b['description']}")
                except:
                    st.info("New badge earned but info unavailable")

            if "clo" in questions[0]:
                st.markdown("### 📚 Learning Outcomes")
                st.markdown(f"- **CLO:** {get_clo_description(questions[0]['clo'])}")
                if "plo" in questions[0]:
                    st.markdown(f"- **PLO:** {get_plo_description(questions[0]['plo'])}")

            if weak_topics:
                st.subheader("📚 Suggested Resources")
                student_level = profile.get("class_level", "matric").lower()
                learning_style = profile.get("learning_style", "visual").lower()

                resources = get_faiss_recommendations(
                    query=f"{chapter} - {topic} for {learning_style} learners",
                    preferred_style=learning_style,
                    level=student_level,
                    selected_topic=topic,
                    k=10
                )

                filtered_resources = [r for r in resources if r.get("url")]

                if filtered_resources:
                    for r in filtered_resources[:5]:
                        duration = round(float(r.get("duration", 0)))
                        st.markdown(f"### 🔹 {r.get('title', 'Untitled')}")
                        st.markdown(f"📄 *{(r.get('description') or 'No description available.').strip()}*")
                        st.markdown(
                            f"🌟 **Format:** {r.get('format', 'N/A')} | "
                            f"🧠 **Difficulty:** {r.get('difficulty', 'N/A')} | "
                            f"⏱ **Duration:** {duration} min"
                        )
                        st.markdown(f"[🔗 Open Resource]({r.get('url', '#')})")
                        st.markdown("---")
                else:
                    st.warning("No personalized matches found. Keep exploring!")

            del st.session_state.practice_questions
            del st.session_state.practice_meta
            del st.session_state.quiz_start_time
