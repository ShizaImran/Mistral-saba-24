import json
import streamlit as st

@st.cache_data
def get_curriculum_data():
    try:
        with open("utils/clos.json", "r") as f:
            clos = json.load(f)
            if not isinstance(clos, list):
                clos = []

        chapters = {
            1: "Binary Systems and Hexadecimal",
            2: "Communication and Internet Technologies",
            3: "Logic Gates and Logic Circuits",
            4: "Operating Systems and Computer Architecture",
            5: "Input and Output Devices",
            6: "Memory and Data Storage",
            7: "High- and Low-Level Languages",
            8: "Security and Ethics"
        }

        curriculum = {"chapters": {}}
        
        for chapter_num, chapter_name in chapters.items():
            chapter_key = f"Chapter {chapter_num}: {chapter_name}"
            curriculum["chapters"][chapter_key] = {
                "clos": [clo for clo in clos if clo.get("chapter") == chapter_num],
                "topics": _get_chapter_topics(chapter_num, clos)
            }

        return curriculum

    except Exception as e:
        st.error(f"❌ Curriculum loading error: {str(e)}")
        return {"chapters": {}}

def _get_chapter_topics(chapter_num, clos):
    topics_map = {
        1: ["Binary Systems", "Hexadecimal", "Number Conversions"],
        2: ["Data Transmission", "Error Detection", "Internet Technologies"],
        3: ["Logic Gates", "Truth Tables", "Boolean Algebra"],
        4: ["OS Functions", "CPU Architecture", "Interrupts"],
        5: ["Input Devices", "Output Devices", "Sensors"],
        6: ["Memory Types", "File Formats", "Storage"],
        7: ["Programming Languages", "Translators", "Debugging"],
        8: ["Security Threats", "Encryption", "Computer Ethics"]
    }

    bloom_levels = {
        "Binary Systems": "Understand",
        "Hexadecimal": "Apply",
        "Number Conversions": "Apply",
        "OS Functions": "Understand",
        "CPU Architecture": "Understand",
        "Interrupts": "Apply",
        # Add more if needed...
    }

    topic_data = {}
    for topic in topics_map.get(chapter_num, []):
        related_clos = [clo["id"] for clo in clos if topic.lower() in clo["description"].lower()]
        topic_data[topic] = {
            "bloom": bloom_levels.get(topic, "Understand"),
            "clos": related_clos
        }

    return topic_data