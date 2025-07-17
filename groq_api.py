import os
import json
from dotenv import load_dotenv
from groq import Groq
from utils.rag_utils import retrieve_context
import traceback

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅🟡 INITIAL QUIZ (Profile Setup)
def generate_initial_quiz(profile):
    try:
        level = profile["class_level"]
        style = profile["learning_style"]
        domain = profile["domain"]
        topic = profile["topic"]

        prompt = f"""
You are a smart AI that creates personalized student quizzes.

Student Profile:
- Level: {level}
- Learning Style: {style}
- Domain: {domain}
- Topic: {topic}

🧠 Instructions:
- Generate 10 MCQs for this student.
- Each question must have 4 options (A, B, C, D).
- Return only VALID JSON in this format:
- Each question object must include a `topic` field reflecting the specific subtopic it's testing.

[
  {{
    "question": "What is AI?",
    "options": ["Artificial Intelligence", "Animal", "Ice", "Apple"],
    "answer": "A"
  }},
  ...
]
No explanations. No markdown. No code fences.
"""

        response = client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You generate JSON-based initial quizzes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        content = response.choices[0].message.content.strip()
        # Remove extra markdown/code fence if exists
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        # Try parsing
        parsed = json.loads(content)

        if not isinstance(parsed, list) or len(parsed) < 5:
            raise ValueError("❌ Invalid or incomplete quiz received")

        return parsed

    except Exception as e:
        print("❌ Initial Quiz JSON Error:", e)
        return [{
            "question": "Fallback: Quiz Generation failed. What do you want's to do now?",
            "options": ["Brather, if you want your son", "high level wrestle", "Send him 2-3 years Dagestan", "& forgot"],
            "answer": "C"
        }]


# ✅🟡 PRACTICE QUIZ (RAG-Based)
def generate_practice_quiz(profile, chapter, topic):
    import traceback
    try:
        level = profile.get("class_level", "Matric")
        domain = profile.get("domain", "Computer Science")

        from utils.rag_utils import retrieve_context
        context = retrieve_context(chapter, topic)

        prompt = f"""
You are an AI that generates student quizzes from textbooks.

🧠 Context:
\"\"\"
{context}
\"\"\"

🎓 Student Info:
- Class Level: {level}
- Domain: {domain}
- Focus Topic: {topic}

Instructions:
- Create 15 MCQs strictly from the provided context.
- Each question must have 4 options (A, B, C, D).
- Each question object must include:
  - "question"
  - "options"
  - "answer"
  - "topic": "{topic}"

Output must be a valid JSON list:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "topic": "{topic}"
  }},
  ...
]
Do not include explanations, markdown, or code fences.
"""

        response = client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You return clean JSON quizzes from textbook context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("json"):
            content = content.replace("json", "").replace("```", "").strip()

        parsed = json.loads(content)

        if isinstance(parsed, list) and len(parsed) >= 5:
            return parsed
        else:
            raise ValueError("Response did not contain enough valid questions.")

    except Exception as e:
        print("❌ PRACTICE QUIZ FALLBACK TRIGGERED")
        print(traceback.format_exc())
        return [{
            "question": "⚠ Error: Quiz generation failed. Retry?",
            "options": ["Yes", "No", "Maybe", "Contact Support"],
            "answer": "A",
            "topic": topic
        }]
    
