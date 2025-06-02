"""
Utility functions for loading and modifying the base models (NLLB, M2M100).
"""
import os
import yaml
from typing import Dict, Any, Optional, Union, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    NllbTokenizer,
    M2M100Tokenizer
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_model_and_tokenizer(
    model_type: str,
    peft_config: Optional[Dict[str, Any]] = None,
    peft_model_path: Optional[str] = None
) -> tuple:
    """
    Load base model and tokenizer with optional PEFT configuration.
    
    Args:
        model_type: Either "nllb" or "m2m100"
        peft_config: Dictionary containing PEFT configuration
        peft_model_path: Path to a saved PEFT model to load
        
    Returns:
        tuple: (model, tokenizer)
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    config = load_config(config_path)
    
    if model_type == "nllb":
        model_name = config["models"]["nllb"]["model_name"]
    elif model_type == "m2m100":
        model_name = config["models"]["m2m100"]["model_name"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load the base model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply PEFT/LoRA if specified
    if peft_config is not None:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("lora_alpha", 16),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            bias=peft_config.get("bias", "none"),
            target_modules=peft_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load saved PEFT model if specified
    if peft_model_path is not None:
        model = PeftModel.from_pretrained(model, peft_model_path)
    
    return model, tokenizer

def get_language_codes(model_type: str) -> Dict[str, str]:
    """Get language codes for the specified model type."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    config = load_config(config_path)
    
    if model_type == "nllb":
        return config["models"]["nllb"]["language_codes"]
    elif model_type == "m2m100":
        return config["models"]["m2m100"]["language_codes"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def translate(
    model: Union[PeftModel, AutoModelForSeq2SeqLM],
    tokenizer: Union[NllbTokenizer, M2M100Tokenizer],
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128,
    num_beams: int = 5
) -> str:
    """
    Translate text from source language to target language.
    
    Args:
        model: Translation model
        tokenizer: Model tokenizer
        text: Text to translate
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum length of generated translation
        num_beams: Number of beams for beam search
        
    Returns:
        str: Translated text
    """
    # Set source and target language for tokenizer
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    # Encode input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode the generated translation
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return translation
