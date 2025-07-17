from firebase_init import db
from datetime import datetime
import streamlit as st

def publish_material(content):
    """Publish content to students"""
    doc_ref = db.collection("published_content").document()
    doc_ref.set({
        **content,
        "published_at": datetime.now(),
        "teacher": st.session_state["email"],
        "view_count": 0
    })

def get_resources_by_topic(topic):
    """Get all resources for a specific topic"""
    docs = db.collection("resources")\
            .where("topics", "array_contains", topic)\
            .stream()
    return [doc.to_dict() for doc in docs]

def get_questions_by_topic(topic, question_types):
    """Get questions filtered by type only (no Bloom's level)"""
    docs = db.collection("questions")\
            .where("topic", "==", topic)\
            .stream()
    all_questions = [doc.to_dict() for doc in docs]
    
    # Filter only by question type if specified
    if question_types:
        return [q for q in all_questions if q["type"] in question_types]
    return all_questions