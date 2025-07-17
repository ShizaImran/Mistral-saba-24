# main.py
import streamlit as st
from auth import login_page
from profile import profile_page
from quiz import quiz_page
from dashboard import dashboard_page
from practice_quiz import practice_quiz_page
from recommender import recommendation_page
from teacher_dashboard import teacher_dashboard
from teacher_profile import teacher_profile_page
from assessment_gen import assessment_generator_page
from lecture_planner import lecture_planner_page
from learning_plan import learning_plan_page

def show_navigation():
    with st.sidebar:
        if st.session_state.page not in ["login", "dashboard", "teacher_dashboard"]:
            if st.button("⬅️ Go Back"):
                go_back()
        
        if st.session_state.page != "login":
            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

if "page_history" not in st.session_state:
    st.session_state.page_history = []

if "page" not in st.session_state:
    st.session_state.page = "login"

if "previous_page" not in st.session_state or st.session_state.previous_page != st.session_state.page:
    st.session_state.page_history.append(st.session_state.page)
    st.session_state.previous_page = st.session_state.page

def go_back():
    if len(st.session_state.page_history) > 1:
        st.session_state.page = st.session_state.page_history[-2]
        st.session_state.page_history = st.session_state.page_history[:-1]
        st.rerun()

show_navigation()

# Page Router
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "profile":
    profile_page()
elif st.session_state.page == "quiz":
    quiz_page()
elif st.session_state.page == "dashboard":
    from firebase_init import db
    if not st.session_state.get("quiz_verified"):
        email = st.session_state.get("email")
        doc = db.collection("students").document(email).get()
        quiz_done = doc.to_dict().get("quiz_done", False)
        if not quiz_done:
            st.session_state.page = "quiz"
            st.session_state.quiz_verified = True
            st.rerun()
        else:
            st.session_state.quiz_verified = True
            dashboard_page()
    else:
        dashboard_page()
elif st.session_state.page == "practice_quiz":
    practice_quiz_page()
elif st.session_state.page == "recommender":
    recommendation_page()
elif st.session_state.page == "teacher_dashboard":
    teacher_dashboard()
elif st.session_state.page == "teacher_profile":
    teacher_profile_page()
elif st.session_state.page == "assessment_generator":
    assessment_generator_page()
elif st.session_state.page == "lecture_planner":
    lecture_planner_page()
elif st.session_state.page == "learning_plan":  # Add this condition
    learning_plan_page()
# Modify your page router
