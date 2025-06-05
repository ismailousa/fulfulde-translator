import os
from datetime import datetime
from utils.jsonl_utils import append_jsonl

def log_correction(original, translation, correction, meta=None):
    """
    Log a user correction to the translations.
    
    Args:
        original: The original input text
        translation: The model-generated translation
        correction: The user-corrected translation
        meta: Additional metadata (source/target language, timestamp, etc.)
    """
    if not meta:
        meta = {}
        
    # Add timestamp if not present
    if "timestamp" not in meta:
        meta["timestamp"] = datetime.now().isoformat()
        
    entry = {
        "original": original,
        "model_translation": translation,
        "user_correction": correction,
        "meta": meta
    }
    
    # Ensure the data directory exists
    os.makedirs("streamlit_ui/data", exist_ok=True)
    
    # Log to JSONL file
    append_jsonl("streamlit_ui/data/corrections.jsonl", entry)
    
    # Return the entry for confirmation
    return entry
