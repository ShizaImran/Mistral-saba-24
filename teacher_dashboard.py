import streamlit as st
from lecture_planner import lecture_planner_page
from assessment_gen import assessment_generator_page
from utils.keygen import generate_widget_key

def teacher_dashboard():
    st.title("👩‍🏫 Teacher Dashboard")

    # Safely define unique radio button key
    tab_selector_key = generate_widget_key("teacher_dashboard", "tab_selector", "static")

    # Use radio buttons to switch between tools
    selected_tab = st.radio(
        "Select a tool:",
        ["📚 Lecture Planner", "📝 Assessment Generator"],
        key=tab_selector_key,
        horizontal=True
    )

    if selected_tab == "📚 Lecture Planner":
        lecture_planner_page(tab_key="lecture_tab")

    elif selected_tab == "📝 Assessment Generator":
        assessment_generator_page(tab_key="assessment_tab")
