# ✅ utils/keygen.py — FINAL WORKING VERSION

def generate_widget_key(page: str, element: str, tab_key: str = "default") -> str:
    """
    Generates a globally unique key for Streamlit widgets.
    Combines page name, element name, and optional tab identifier.
    Replaces spaces with underscores and lowers the case.
    """
    return f"{page}_{tab_key}_{element}".replace(" ", "_").lower()
