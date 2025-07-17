import streamlit as st
from firebase_init import db, get_progress_ref, update_overall_stats
from datetime import datetime

# ✅ Initial Quiz Evaluation (One-time Assessment)
def evaluate_initial_quiz(questions, answers):
    score = 0
    corrects = []

    for q, ans in zip(questions, answers):
        correct = q["answer"]
        corrects.append(correct)
        if ans == correct:
            score += 1

    if score <= 2:
        level = "Beginner"
    elif score == 3:
        level = "Intermediate"
    elif score == 4:
        level = "Advanced"
    else:
        level = "Expert"

    email = st.session_state.get("email")
    if email:
        db.collection("students").document(email).update({
            "quiz_done": True,
            "quiz_score": score,
            "result_level": level,
            "last_active": datetime.now()
        })

    return score, level, corrects

# ✅ Practice Quiz Evaluation
def evaluate_practice_quiz(questions, answers):
    score = 0
    weak_topics = []

    for q, ans in zip(questions, answers):
        correct = q["answer"]
        if ans == correct:
            score += 1
        else:
            topic = q.get("topic", "").strip()
            if topic and topic.lower() != "unknown":
                weak_topics.append(topic)

    percent = round((score / len(questions)) * 100, 2)
    return percent, weak_topics


# ✅ Subtopic Performance Tracking + Attempts Logging
def update_subtopic_progress(email, percent, topic, chapter, duration):
    progress_ref = get_progress_ref(email)
    topic_id = topic.lower().replace(" ", "_")
    topic_doc_ref = progress_ref.collection("topics").document(topic_id)

    doc = topic_doc_ref.get()
    data = doc.to_dict() if doc.exists else {}

    scores = data.get("scores", [])
    scores.append(percent)

    new_status = get_status(percent)

    topic_doc_ref.set({
        "topic": topic,
        "chapter": chapter,
        "topic_id": topic_id,
        "status": new_status,
        "scores": scores,
        "last_score": percent,
        "best_score": max(scores),
        "total_attempts": len(scores),
        "last_attempted": datetime.now(),
        "last_duration": duration
    }, merge=True)

    update_overall_stats(email)
    update_summary_weak_topics(email)  # 🔁 new summary logic

# ✅ Determine Status from Score
def get_status(score):
    if score < 40:
        return "Weak"
    elif score < 65:
        return "Needs Improvement"
    elif score < 85:
        return "Moderate"
    else:
        return "Mastered"

# ✅ Summary Generator for Weak Topics (Single Document Version)
def update_summary_weak_topics(email):
    try:
        progress_ref = get_progress_ref(email)
        topic_docs = progress_ref.collection("topics").stream()

        summary = []
        for doc in topic_docs:
            data = doc.to_dict()
            status = data.get("status", "").lower()
            topic_name = data.get("topic", "Unknown")
            scores = data.get("scores", [])

            if status in ["weak", "needs improvement", "moderate"] and scores:
                avg_score = round(sum(scores) / len(scores), 2)
                summary.append({
                    "topic": topic_name,
                    "status": status.capitalize(),
                    "score": avg_score
                })

        progress_ref.collection("weak_topic").document("summary").set({
            "topics": summary,
            "updated_at": datetime.now()
        })

    except Exception as e:
        print("⚠️ Failed to update weak topic summary:", e)
