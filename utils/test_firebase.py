# This code checks if the Firestore connection is working by trying to read a document.
# You should see False or True → ✅ means connected.
from firebase_init import db
print(db.collection("test").document("ping").get().exists)