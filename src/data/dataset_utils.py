"""
Utility functions for loading and preprocessing datasets.
"""
import os
import json
from typing import Dict, List, Tuple, Optional
import yaml

import pandas as pd
from datasets import Dataset, DatasetDict


def load_jsonl_data(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List[Dict]: List of JSON records
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data


def convert_to_translation_format(
    data: List[Dict],
    model_type: str,
    config_path: Optional[str] = None
) -> List[Dict]:
    """
    Convert raw data to translation format.
    
    Args:
        data: List of JSON records (e.g., {"english": "...", "fulfulde": "...", "french": "..."})
        model_type: Either "nllb" or "m2m100"
        config_path: Path to config file
        
    Returns:
        List[Dict]: List of records in translation format (e.g., {"translation": {"ff": "...", "en": "...", "fr": "..."}})
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == "nllb":
        lang_codes = config["models"]["nllb"]["language_codes"]
    elif model_type == "m2m100":
        lang_codes = config["models"]["m2m100"]["language_codes"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    translation_data = []
    for item in data:
        translation_entry = {"translation": {}}
        
        # Map "english", "fulfulde", "french" to the appropriate language codes
        if "english" in item and item["english"]:
            translation_entry["translation"][lang_codes["english"]] = item["english"]
        
        if "fulfulde" in item and item["fulfulde"]:
            translation_entry["translation"][lang_codes["fulfulde"]] = item["fulfulde"]
        
        if "french" in item and item["french"]:
            translation_entry["translation"][lang_codes["french"]] = item["french"]
        
        translation_data.append(translation_entry)
    
    return translation_data


def create_translation_pair_dataset(
    data: List[Dict],
    src_lang: str,
    tgt_lang: str,
    test_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Create a dataset for a specific translation direction.
    
    Args:
        data: List of records in translation format
        src_lang: Source language code
        tgt_lang: Target language code
        test_size: Proportion of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict: HuggingFace dataset with train and test splits
    """
    # Filter out entries that don't have both source and target languages
    filtered_data = []
    for item in data:
        if src_lang in item["translation"] and tgt_lang in item["translation"]:
            filtered_data.append({
                "source": item["translation"][src_lang],
                "target": item["translation"][tgt_lang],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            })
    
    # Create DataFrame and convert to Dataset
    df = pd.DataFrame(filtered_data)
    dataset = Dataset.from_pandas(df)
    
    # Split into train and test
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    
    return DatasetDict({
        "train": splits["train"],
        "test": splits["test"]
    })


def process_dataset_for_model(
    jsonl_path: str,
    model_type: str,
    src_lang: str,
    tgt_lang: str,
    test_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Process a JSONL file into a dataset for a specific translation direction.
    
    Args:
        jsonl_path: Path to the JSONL file
        model_type: Either "nllb" or "m2m100"
        src_lang: Source language code
        tgt_lang: Target language code
        test_size: Proportion of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict: HuggingFace dataset with train and test splits
    """
    # Load raw data
    raw_data = load_jsonl_data(jsonl_path)
    
    # Convert to translation format
    translation_data = convert_to_translation_format(raw_data, model_type)
    
    # Create dataset for the specific translation direction
    dataset = create_translation_pair_dataset(
        translation_data,
        src_lang,
        tgt_lang,
        test_size=test_size,
        seed=seed
    )
    
    return dataset
