import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

clo_map = load_json("utils/clos.json")
bloom_map = load_json("utils/bloom_level.json")
study_resources = load_json("utils/cleaned_study_resources(2).json")

def get_clos_for_topic(topic):
    return [c["id"] for c in clo_map if topic.lower() in c["description"].lower()]

def get_bloom_level_for_topic(topic):
    return bloom_map.get(topic, "Understand")

def get_study_resources(topic):
    return [res for res in study_resources if topic.lower() in res["topic"].lower()]
def get_plos_for_topic(topic):
    # Simulated mapping for PLOs
    return ["PLO1: Communication", "PLO3: Problem Solving"]

def get_instructor_profile():
    return {
        "style": "Visual + Hands-On",
        "tips": "Use diagrams, real-world analogies, and interactive tasks to reinforce concepts."
    }

