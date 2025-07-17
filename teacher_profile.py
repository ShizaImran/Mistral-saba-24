import streamlit as st
from firebase_init import db
from firebase_admin import firestore  # Add this import

def teacher_profile_page():
    st.title("📝 Teacher Profile")
    
    email = st.session_state.get("email")
    if not email:
        st.error("⚠️ Session expired. Please log in again.")
        return

    doc_ref = db.collection("teachers").document(email)
    doc = doc_ref.get()

    if not doc.exists:
        st.error("❌ Teacher profile not found.")
        return

    teacher = doc.to_dict()

    with st.form("profile_form"):
        st.subheader("Basic Information")
        name = st.text_input("Full Name", value=teacher.get("name", ""))
        subject = st.text_input("Primary Subject", value=teacher.get("subject", ""))
        
        st.subheader("Teaching Preferences")
        experience = st.selectbox(
            "Experience Level",
            ["1-3 years", "4-7 years", "8+ years"],
            index=["1-3 years", "4-7 years", "8+ years"].index(
                teacher.get("experience", "1-3 years"))
        )
        style = st.selectbox(
            "Preferred Teaching Style",
            ["Lecture", "Interactive", "Demonstration", "Discussion"],
            index=["Lecture", "Interactive", "Demonstration", "Discussion"].index(
                teacher.get("teaching_style", "Lecture"))
        )

        if st.form_submit_button("Save Profile"):
            updates = {
                "name": name,
                "subject": subject,
                "experience": experience,
                "teaching_style": style,
                "last_updated": firestore.SERVER_TIMESTAMP  # Fixed this line
            }
            doc_ref.update(updates)
            st.success("Profile updated successfully!")

    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = "teacher_dashboard"
        st.rerun()