# utils/resource_cleaner.py

import pandas as pd

def clean_learning_dataset(path="utils\RS_Dataset3.xlsx", save_as="cleaned_resources.json"):
    # Load Excel with multi-level headers
    df = pd.read_excel(path, header=[0, 1])

    # Drop third row if it's placeholder (unnamed or blank)
    df.columns = [f"{lvl1.strip()}.{lvl2.strip()}" if isinstance(lvl1, str) and isinstance(lvl2, str) else lvl2 for lvl1, lvl2 in df.columns]

    # Drop empty/NaN columns
    df = df.dropna(axis=1, how="all")

    # Select only relevant features
    selected_cols = [
        "General.Identifier",
        "General.Title",
        "General.Description",
        "Annotation.Focused Summary",
        "Classification.Topic",
        "Classification.Chapter",
        "Classification.Subtopic",
        "Classification.Learning Objective",
        "Classification.Bloom Taxonomy Level",
        "Educational.Difficulty",
        "Educational.Typical Learning Time",
        "Educational.Learning Resource Type",
        "Technical.Format",
        "Technical.Location",
        "Classification.TopicCoverageRatio",
        "Classification.Purpose"
    ]

    # Filter available columns from Excel
    filtered_cols = [col for col in selected_cols if col in df.columns]
    df_filtered = df[filtered_cols].dropna(subset=["General.Title", "Technical.Location"], how="any")

    # Save to JSON
    df_filtered.to_json(save_as, orient="records", force_ascii=False, indent=2)
    print(f"✅ Cleaned {len(df_filtered)} resources saved to: {save_as}")

# Run directly
if __name__ == "__main__":
    clean_learning_dataset()
