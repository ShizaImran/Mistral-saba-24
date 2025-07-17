import streamlit as st
from firebase_init import db
from groq_api import generate_initial_quiz
from evaluate import evaluate_initial_quiz
import time

def quiz_page():
    st.title("📝 Quiz Assessment")

    email = st.session_state.get("email")
    profile = st.session_state.get("profile")

    # Get quiz if not already generated
    if "quiz_questions" not in st.session_state:
        if not profile:
            doc = db.collection("students").document(email).get()
            profile = doc.to_dict()
            st.session_state.profile = profile
        questions = generate_initial_quiz(profile)
        st.session_state.quiz_questions = questions

    questions = st.session_state.quiz_questions
    #st.write("📋 Loaded Quiz:", questions)  
    if not questions:
        st.error("❌ Failed to load quiz questions. Please try again later.")
        return
    answers = []

    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}: {q.get('question', 'No question')}**")

        options = q.get("options", [])
        choice = st.radio("Choose an option:", options, key=f"q{i}")

        # Convert option text to A/B/C/D
        if choice in options:
            index = options.index(choice)
            selected_letter = chr(65 + index)  # 65 = A
            answers.append(selected_letter)
        else:
            answers.append("-")  # fallback



    if st.button("Submit Quiz"):
        score, level, corrects = evaluate_initial_quiz(questions, answers)

        db.collection("students").document(email).update({
            "quiz_score": score,
            "result_level": level,
            "answers": answers,
            "correct_answers": corrects,
            "quiz_done": True
        })

        time.sleep(1)  # 🛑 Give Firebase time to sync

        st.success("✅ Quiz submitted. Your profile is now complete.")
        st.session_state.page = "dashboard"
        st.session_state.quiz_verified = True  # ✅ prevent rerun loop
        st.rerun()

