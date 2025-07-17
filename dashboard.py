import streamlit as st
from firebase_init import db
from datetime import datetime
import pandas as pd
import plotly.express as px

# Custom CSS for styling
st.markdown("""
<style>
    .header-style {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        background-color: #e9ecef;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        background-color: #4e73df;
    }
    .btn-primary {
        background-color: #4e73df !important;
        color: white !important;
        border: none !important;
    }
    .btn-secondary {
        background-color: #858796 !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

def dashboard_page():
    # Page header with logo
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://via.placeholder.com/100", width=80)  # Replace with your logo
    with col2:
        st.title("Student Learning Dashboard")
    
    # Authentication check
    email = st.session_state.get("email")
    if not email:
        st.error("🔒 Session expired. Please log in again.")
        return

    student = db.collection("students").document(email).get().to_dict()
    if not student:
        st.error("❌ Student record not found.")
        return

    # Main dashboard layout
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Progress", "⚡ Quick Actions"])

    with tab1:
        # Profile card
        with st.container():
            st.subheader("👤 Student Profile")
            cols = st.columns(2)
            
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="header-style">Personal Info</div>
                    <p><strong>Name:</strong> {student.get('name', 'N/A')}</p>
                    <p><strong>Email:</strong> {student.get('email', 'N/A')}</p>
                    <p><strong>Last Active:</strong> {student.get('last_active', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="header-style">Academic Info</div>
                    <p><strong>Level:</strong> {student.get('class_level', 'N/A')}</p>
                    <p><strong>Domain:</strong> {student.get('domain', 'N/A')}</p>
                    <p><strong>Focus Topic:</strong> {student.get('topic', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

        # Performance metrics
        st.subheader("📌 Performance Summary")
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="header-style">Initial Assessment</div>
                <h3>{student.get('quiz_score', 0)}/5</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {student.get('quiz_score', 0)*20}%"></div>
                </div>
                <p>Proficiency: {student.get('result_level', 'Beginner')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            # Example - would need actual progress data
            avg_score = 72  # Replace with real calculation
            st.markdown(f"""
            <div class="metric-card">
                <div class="header-style">Average Score</div>
                <h3>{avg_score}%</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {avg_score}%"></div>
                </div>
                <p>Across all topics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            # Example - would need actual data
            mastered = 3  # Replace with real data
            st.markdown(f"""
            <div class="metric-card">
                <div class="header-style">Mastered Topics</div>
                <h3>{mastered}</h3>
                <p>Out of 8 core topics</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        # Progress visualization
        st.subheader("📈 Learning Progress")
        
        # Example data - replace with actual progress data
        progress_data = {
            "Topic": ["Binary Systems", "Logic Gates", "Operating Systems", "Memory"],
            "Last Score": [85, 72, 68, 91],
            "Status": ["Strong", "Moderate", "Weak", "Mastered"]
        }
        df = pd.DataFrame(progress_data)
        
        # Color mapping for status
        color_discrete_map = {
            "Weak": "#e74a3b",
            "Moderate": "#f6c23e",
            "Strong": "#1cc88a",
            "Mastered": "#4e73df"
        }
        
        fig = px.bar(df, 
                    x="Topic", 
                    y="Last Score",
                    color="Status",
                    color_discrete_map=color_discrete_map,
                    text="Last Score",
                    height=400)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Quick actions with improved buttons
        st.subheader("⚡ Quick Access")
        
        cols = st.columns(2)
        with cols[0]:
            if st.button("🧪 Start Practice Quiz", 
                        key="practice_quiz",
                        help="Test your knowledge on current topics",
                        use_container_width=True):
                st.session_state.page = "practice_quiz"
                st.rerun()
            
            if st.button("📚 View Recommendations", 
                        key="recommendations",
                        help="Personalized learning resources",
                        use_container_width=True):
                st.session_state.page = "recommender"
                st.rerun()
        
        with cols[1]:
            if st.button("📝 Learning Pan", 
                        key="learning_plan",
                        help="Your personalized learning plan",
                        use_container_width=True):
                st.session_state.page = "learning_plan"
                st.rerun()
            
            if st.button("📊 View Teacher Materials", 
                        key="teacher_content",
                        help="see materials shared by your teacher",
                        use_container_width=True):
                st.session_state.page = "teacher_content"
                st.rerun()
        
        st.markdown("---")
        st.markdown("#### 🎯 Current Focus Area")
        st.markdown(f"**{student.get('topic', 'No topic selected')}**")
        
        # Example learning plan progress
        st.markdown("""
        <div class="metric-card">
            <div class="header-style">Learning Plan Progress</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 45%"></div>
            </div>
            <p>45% completed (3/7 activities)</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard_page()