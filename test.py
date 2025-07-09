from faiss_search import search_resources

# Simulate a weak topic
weak_topic = "Binary to Denary Conversion"

# Get recommendations
results = search_resources(weak_topic)

for r in results:
    print("🔹", r["title"])
    print("🔗", r["url"])
    print("📘", r.get("description", "No description."))
    print("---")