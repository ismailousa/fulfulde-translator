"""
Utility functions for tokenization and special token handling.
"""
from typing import Dict, List, Optional
import os
import yaml

from transformers import PreTrainedTokenizer, AutoTokenizer


def get_tokenizer(model_type: str) -> PreTrainedTokenizer:
    """
    Get the appropriate tokenizer for the model type.
    
    Args:
        model_type: Either "nllb" or "m2m100"
        
    Returns:
        PreTrainedTokenizer: The model tokenizer
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if model_type == "nllb":
        model_name = config["models"]["nllb"]["model_name"]
    elif model_type == "m2m100":
        model_name = config["models"]["m2m100"]["model_name"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return AutoTokenizer.from_pretrained(model_name)


def get_language_ids(tokenizer: PreTrainedTokenizer, lang_codes: List[str]) -> Dict[str, int]:
    """
    Get language IDs for the specified language codes.
    
    Args:
        tokenizer: Model tokenizer
        lang_codes: List of language codes
        
    Returns:
        Dict[str, int]: Mapping of language codes to IDs
    """
    lang_ids = {}
    for lang_code in lang_codes:
        try:
            lang_ids[lang_code] = tokenizer.lang_code_to_id[lang_code]
        except KeyError:
            print(f"Warning: Language code '{lang_code}' not found in tokenizer vocabulary.")
    
    return lang_ids


def set_language_pair(tokenizer: PreTrainedTokenizer, src_lang: str, tgt_lang: str) -> PreTrainedTokenizer:
    """
    Set source and target language for tokenizer.
    
    Args:
        tokenizer: Model tokenizer
        src_lang: Source language code
        tgt_lang: Target language code
        
    Returns:
        PreTrainedTokenizer: Configured tokenizer
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    return tokenizer


def preprocess_text(text: str) -> str:
    """
    Normalize text for translation.
    
    Args:
        text: Input text
        
    Returns:
        str: Normalized text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Normalize punctuation spacing
    for punct in ".,;:!?()[]{}\"'":
        text = text.replace(f" {punct}", punct)
    
    return text


def batch_tokenize(
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    src_lang: str,
    tgt_lang: Optional[str] = None,
    max_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True
) -> Dict:
    """
    Tokenize a batch of texts.
    
    Args:
        tokenizer: Model tokenizer
        texts: List of input texts
        src_lang: Source language code
        tgt_lang: Target language code (optional)
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate sequences
        
    Returns:
        Dict: Tokenized inputs
    """
    # Set language for tokenizer
    tokenizer.src_lang = src_lang
    if tgt_lang:
        tokenizer.tgt_lang = tgt_lang
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Tokenize
    inputs = tokenizer(
        processed_texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt"
    )
    
    return inputs
