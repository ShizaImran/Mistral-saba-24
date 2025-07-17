import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
    })

db = firestore.client()

# 📦 Modular Ref Helpers
def get_user_ref(email):
    return db.collection("students").document(email)

def get_progress_ref(email):
    return db.collection("progress").document(email)

def update_overall_stats(email):
    progress_ref = get_progress_ref(email)
    topics = progress_ref.collection("topics").stream()

    total = 0
    mastered = 0
    for doc in topics:
        data = doc.to_dict()
        total += 1
        if data.get("status") == "Mastered":
            mastered += 1

    stats = {
        "total_topics": total,
        "mastered_topics": mastered,
        "last_updated": firestore.SERVER_TIMESTAMP
    }
    progress_ref.set({"overall_stats": stats}, merge=True)