import streamlit as st
from firebase_init import db

def login_page():
    st.title("🔐 Login / Sign Up")

    # 👤 Select user type
    user_type = st.selectbox("Login as:", ["Student", "Teacher"])

    # 🔁 Sign In or Sign Up switch
    mode = st.radio("Select Mode", ["Sign In", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Continue"):
        if not email or not password:
            st.error("Email and password are required.")
            return

        # 🔍 Use correct collection based on user type
        collection_name = "students" if user_type == "Student" else "teachers"
        doc_ref = db.collection(collection_name).document(email)
        doc = doc_ref.get()

        st.session_state.email = email
        st.session_state.user_type = user_type

        # 🔐 SIGN IN
        if mode == "Sign In":
            if doc.exists:
                data = doc.to_dict()
                if data.get("password") == password:
                    # Redirect to respective dashboard
                    st.session_state.page = "dashboard" if user_type == "Student" else "teacher_dashboard"
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Account does not exist. Please sign up first.")

        # 🆕 SIGN UP
        elif mode == "Sign Up":
            if doc.exists:
                st.warning("Account already exists. Please sign in.")
            else:
                doc_ref.set({
                    "email": email,
                    "password": password,
                    "quiz_done": False if user_type == "Student" else None
                })
                # Redirect to respective profile setup
                st.session_state.page = "profile" if user_type == "Student" else "teacher_profile"
                st.success("Account created. Proceeding to profile setup...")
                st.rerun()
