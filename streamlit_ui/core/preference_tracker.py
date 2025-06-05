import os
from datetime import datetime
from utils.jsonl_utils import append_jsonl

def log_preference(text, options, selected, meta=None):
    """
    Log user preferences between different translation options.
    
    Args:
        text: The original input text
        options: Dictionary of model_name: translation pairs
        selected: The name of the selected model/variant
        meta: Additional metadata (source/target language, timestamp, etc.)
    """
    if not meta:
        meta = {}
        
    # Add timestamp if not present
    if "timestamp" not in meta:
        meta["timestamp"] = datetime.now().isoformat()
    
    # Format the entry
    entry = {
        "input_text": text,
        "options": options,
        "preferred_model": selected,
        "meta": meta
    }
    
    # Ensure the data directory exists
    os.makedirs("streamlit_ui/data", exist_ok=True)
    
    # Log to JSONL file
    append_jsonl("streamlit_ui/data/preferences.jsonl", entry)
    
    # Return the entry for confirmation
    return entry
