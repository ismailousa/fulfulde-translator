import os
import json
import jsonlines
from typing import List, Dict, Any

def append_jsonl(file_path: str, entry: Dict[str, Any]) -> None:
    """
    Safely append an entry to a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        entry: Dictionary to append
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Append the entry to the JSONL file
    with jsonlines.open(file_path, mode='a') as writer:
        writer.write(entry)

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Safely read entries from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, each representing an entry
    """
    entries = []
    
    # Check if the file exists
    if not os.path.exists(file_path):
        return entries
    
    # Read entries from the JSONL file
    try:
        with jsonlines.open(file_path, mode='r') as reader:
            for entry in reader:
                entries.append(entry)
    except Exception as e:
        print(f"Error reading JSONL file: {str(e)}")
    
    return entries

def export_jsonl_to_csv(jsonl_path: str, csv_path: str) -> bool:
    """
    Export a JSONL file to CSV format.
    
    Args:
        jsonl_path: Path to the JSONL file
        csv_path: Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        # Read the JSONL file
        data = read_jsonl(jsonl_path)
        
        if not data:
            return False
        
        # Convert to DataFrame and export to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {str(e)}")
        return False
