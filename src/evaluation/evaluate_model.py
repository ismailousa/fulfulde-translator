"""
Evaluation script for fine-tuned translation models.
"""
import os
import argparse
import json
from typing import Dict, List, Any, Optional

import torch
import evaluate
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from peft import PeftModel

# Import local modules (if needed)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.model_utils import get_model_and_tokenizer


def load_model_and_tokenizer(model_dir: str, model_type: str, use_peft: bool = True) -> tuple:
    """Load the model and tokenizer."""
    if use_peft:
        # For PEFT models, we need to load the base model first
        if model_type == "nllb":
            base_model_id = "facebook/nllb-200-distilled-600M"
        else:  # m2m100
            base_model_id = "facebook/m2m100_418M"
        
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
        model = PeftModel.from_pretrained(model, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        # For full models, load directly
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer


def load_test_data(data_path: str, src_lang: str, tgt_lang: str) -> List[Dict[str, str]]:
    """Load test data from JSONL file."""
    test_examples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            
            # Identify source and target based on language codes
            if src_lang in example and tgt_lang in example:
                test_examples.append({
                    "source": example[src_lang],
                    "reference": example[tgt_lang]
                })
    
    # Only use a subset for testing if dataset is large
    test_size = min(len(test_examples), 100)
    return test_examples[:test_size]


def translate_text(text: str, model, tokenizer, src_lang: str, tgt_lang: str, 
                   max_length: int = 128) -> str:
    """Translate a text using the model."""
    # Set source language
    tokenizer.src_lang = src_lang
    
    # Determine forced BOS token ID based on tokenizer type
    # NLLB tokenizers don't have lang_code_to_id attribute
    if "nllb" in tokenizer.name_or_path.lower():
        # For NLLB, forced BOS token is in format "__<lang>__"
        forced_bos_token = f"__{tgt_lang}__"
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(forced_bos_token)
    else:  # M2M100
        # Set target language for M2M100
        tokenizer.tgt_lang = tgt_lang
        # M2M100 has get_lang_id method
        forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    
    # Move inputs to the model's device
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length
        )
    
    # Decode output
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translation


def evaluate_model(model_dir: str,
                  data_path: str,
                  model_type: str,
                  src_lang: str,
                  tgt_lang: str,
                  use_peft: bool = True,
                  max_samples: int = 100,
                  show_examples: int = 5) -> Dict[str, Any]:
    """Evaluate the model and return metrics."""
    print(f"Evaluating model from {model_dir}...")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir, model_type, use_peft)
    
    # Move model to appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load test data
    test_data = load_test_data(data_path, src_lang, tgt_lang)
    if max_samples and len(test_data) > max_samples:
        test_data = test_data[:max_samples]
    print(f"Evaluating on {len(test_data)} examples")
    
    # Load metrics
    bleu = evaluate.load("sacrebleu")
    
    # Translate and compute metrics
    translations = []
    references = []
    examples = []
    
    for i, example in enumerate(tqdm(test_data)):
        source_text = example["source"]
        reference_text = example["reference"]
        
        # Translate
        translation = translate_text(
            source_text, model, tokenizer, src_lang, tgt_lang
        )
        
        # Save results
        translations.append(translation)
        references.append([reference_text])  # BLEU expects a list of references
        
        # Save examples for display
        if i < show_examples:
            examples.append({
                "source": source_text,
                "translation": translation,
                "reference": reference_text
            })
    
    # Compute metrics
    bleu_results = bleu.compute(predictions=translations, references=references)
    
    # Print results
    print(f"\nBLEU score: {bleu_results['score']:.2f}")
    print("\nExample translations:")
    print("-" * 80)
    
    # Show examples
    for i, example in enumerate(examples):
        print(f"Example {i+1}:")
        print(f"Source ({src_lang}): {example['source']}")
        print(f"Translation ({tgt_lang}): {example['translation']}")
        print(f"Reference ({tgt_lang}): {example['reference']}")
        print("-" * 80)
    
    return {
        "bleu_score": bleu_results["score"],
        "num_examples": len(test_data),
        "examples": examples
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a translation model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with the trained model")
    parser.add_argument("--model_type", type=str, required=True, choices=["nllb", "m2m100"],
                        help="Model type (nllb or m2m100)")
    parser.add_argument("--src_lang", type=str, required=True,
                        help="Source language code")
    parser.add_argument("--tgt_lang", type=str, required=True,
                        help="Target language code")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the JSONL data file")
    parser.add_argument("--use_peft", action="store_true",
                        help="Whether to use PEFT/LoRA")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--show_examples", type=int, default=5,
                        help="Number of example translations to show")
    
    args = parser.parse_args()
    
    evaluate_model(
        args.model_dir,
        args.data_path,
        args.model_type,
        args.src_lang,
        args.tgt_lang,
        args.use_peft,
        args.max_samples,
        args.show_examples
    )
