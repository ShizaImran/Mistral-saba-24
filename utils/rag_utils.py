# 📘 Extracts, chunks, and serves relevant textbook content for RAG-based quiz generation

import fitz  # PyMuPDF
import json
import re
import os

# 📤 Extract and split chapters into meaningful chunks
def extract_chapters_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    raw_text = ""

    for page in doc:
        raw_text += page.get_text()

    chapters = re.split(r"(Chapter\s+\d+[\s\S]*?)\n", raw_text)
    structured = []

    for i in range(1, len(chapters), 2):
        title = chapters[i].strip()
        content = chapters[i + 1].strip()
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]

        for para in paragraphs:
            structured.append({
                "chapter": title,
                "text": para
            })

    return structured

# 💾 Save extracted chunks to a JSON file
def save_chunks_to_json(chunks, out_file="rag_book_chunks.json"):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

# 📂 Load chunks from JSON file
def load_chunks(json_path="rag_book_chunks.json"):
    if not os.path.exists(json_path):
        print("❌ Missing rag_book_chunks.json. Run chunk_book() first.")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 🔍 Retrieve relevant context from textbook
def retrieve_context(chapter, topic, limit=3):
    chunks = load_chunks()
    relevant = [
        chunk["text"]
        for chunk in chunks
        if chapter.lower() in chunk["chapter"].lower() and topic.lower() in chunk["text"].lower()
    ]
    return " ".join(relevant[:limit]) if relevant else ""

# 🚀 Rune this once by run_rag_utils.py : Extract chunks and save to JSON
def chunk_book(pdf_path="your_book.pdf"):
    chunks = extract_chapters_from_pdf(pdf_path)
    save_chunks_to_json(chunks)
    print(f"✅ {len(chunks)} chunks saved to rag_book_chunks.json")
