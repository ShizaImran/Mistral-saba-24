import json
from pathlib import Path

    # Load content data

content_path = Path("utils/cleaned_study_resources(2).json")
with open(content_path, encoding='utf-8') as f:
    content_data = json.load(f)
    print("Content data loaded successfully.")
    print(content_data[:1])  # Print first two items for verification