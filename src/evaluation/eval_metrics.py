"""
Evaluation metrics and utilities for translation models.
"""
import os
from typing import Dict, List, Optional, Union, Tuple
import json
import yaml

import torch
import numpy as np
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer
from peft import PeftModel

# Add the project root to path for absolute imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use absolute import
from src.models.model_utils import translate


def evaluate_bleu(
    model: Union[PeftModel, AutoModelForSeq2SeqLM],
    tokenizer: PreTrainedTokenizer,
    test_data: List[Dict[str, str]],
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Evaluate model using BLEU score.
    
    Args:
        model: Translation model
        tokenizer: Model tokenizer
        test_data: List of test examples with 'source' and 'target' keys
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum sequence length
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    metric = evaluate.load("sacrebleu")
    predictions = []
    references = []
    
    for example in test_data:
        source_text = example["source"]
        target_text = example["target"]
        
        # Generate translation
        translation = translate(
            model, 
            tokenizer, 
            source_text, 
            src_lang, 
            tgt_lang, 
            max_length
        )
        
        predictions.append(translation)
        references.append([target_text])
    
    # Compute BLEU score
    result = metric.compute(predictions=predictions, references=references)
    
    return {
        "bleu": result["score"],
        "count": len(test_data)
    }


def round_trip_evaluation(
    model: Union[PeftModel, AutoModelForSeq2SeqLM],
    tokenizer: PreTrainedTokenizer,
    test_data: List[Dict[str, str]],
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Evaluate model using round-trip translation.
    
    Args:
        model: Translation model
        tokenizer: Model tokenizer
        test_data: List of test examples with 'source' key
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum sequence length
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    metric = evaluate.load("sacrebleu")
    predictions = []
    references = []
    
    for example in test_data:
        source_text = example["source"]
        
        # Forward translation (src -> tgt)
        forward_translation = translate(
            model, 
            tokenizer, 
            source_text, 
            src_lang, 
            tgt_lang, 
            max_length
        )
        
        # Backward translation (tgt -> src)
        backward_translation = translate(
            model, 
            tokenizer, 
            forward_translation, 
            tgt_lang, 
            src_lang, 
            max_length
        )
        
        predictions.append(backward_translation)
        references.append([source_text])
    
    # Compute BLEU score for round-trip translation
    result = metric.compute(predictions=predictions, references=references)
    
    return {
        "round_trip_bleu": result["score"],
        "count": len(test_data)
    }


def analyze_errors(
    model: Union[PeftModel, AutoModelForSeq2SeqLM],
    tokenizer: PreTrainedTokenizer,
    test_data: List[Dict[str, str]],
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128,
    n_worst: int = 10
) -> List[Dict[str, str]]:
    """
    Analyze worst translations based on BLEU score.
    
    Args:
        model: Translation model
        tokenizer: Model tokenizer
        test_data: List of test examples with 'source' and 'target' keys
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum sequence length
        n_worst: Number of worst translations to return
        
    Returns:
        List[Dict[str, str]]: Analysis of worst translations
    """
    metric = evaluate.load("sacrebleu")
    results = []
    
    for example in test_data:
        source_text = example["source"]
        target_text = example["target"]
        
        # Generate translation
        translation = translate(
            model, 
            tokenizer, 
            source_text, 
            src_lang, 
            tgt_lang, 
            max_length
        )
        
        # Compute BLEU score for this example
        score = metric.compute(predictions=[translation], references=[[target_text]])["score"]
        
        results.append({
            "source": source_text,
            "target": target_text,
            "prediction": translation,
            "bleu": score
        })
    
    # Sort by BLEU score (ascending) and get the worst n translations
    worst_translations = sorted(results, key=lambda x: x["bleu"])[:n_worst]
    
    return worst_translations


def save_evaluation_results(
    results: Dict[str, Union[float, List[Dict[str, str]]]],
    output_file: str
):
    """
    Save evaluation results to a file.
    
    Args:
        results: Evaluation results
        output_file: Output file path
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_full_evaluation(
    model_path: str,
    test_data_path: str,
    src_lang: str,
    tgt_lang: str,
    model_type: str = "nllb",
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    is_peft_model: bool = True
) -> Dict[str, Union[float, List[Dict[str, str]]]]:
    """
    Run a full evaluation suite on a trained model.
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test data (JSON)
        src_lang: Source language code
        tgt_lang: Target language code
        model_type: Model type ("nllb" or "m2m100")
        output_dir: Directory to save evaluation results
        config_path: Path to the configuration file
        is_peft_model: Whether the model is a PEFT model
        
    Returns:
        Dict[str, Union[float, List[Dict[str, str]]]]: Evaluation results
    """
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load base model and tokenizer
    if model_type == "nllb":
        base_model_name = config["models"]["nllb"]["model_name"]
    elif model_type == "m2m100":
        base_model_name = config["models"]["m2m100"]["model_name"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model and tokenizer
    if is_peft_model:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load test data
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Run evaluations
    bleu_results = evaluate_bleu(
        model,
        tokenizer,
        test_data,
        src_lang,
        tgt_lang,
        max_length=config["data"]["max_length"]
    )
    
    round_trip_results = round_trip_evaluation(
        model,
        tokenizer,
        test_data,
        src_lang,
        tgt_lang,
        max_length=config["data"]["max_length"]
    )
    
    error_analysis = analyze_errors(
        model,
        tokenizer,
        test_data,
        src_lang,
        tgt_lang,
        max_length=config["data"]["max_length"]
    )
    
    # Combine results
    results = {
        "model_type": model_type,
        "model_path": model_path,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "bleu": bleu_results["bleu"],
        "round_trip_bleu": round_trip_results["round_trip_bleu"],
        "sample_count": bleu_results["count"],
        "error_analysis": error_analysis
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"eval_{src_lang}_to_{tgt_lang}.json")
        save_evaluation_results(results, output_file)
    
    return results
