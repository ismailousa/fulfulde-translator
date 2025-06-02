#!/usr/bin/env python
"""
Example script for using the Fulfulde Translator.
This demonstrates how to load models and perform translations.
"""
import argparse
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_utils import get_model_and_tokenizer, translate, get_language_codes


def main():
    parser = argparse.ArgumentParser(description="Translate text between Fulfulde, English, and French")
    parser.add_argument("--model_type", type=str, default="nllb", choices=["nllb", "m2m100"],
                      help="Model type to use (nllb or m2m100)")
    parser.add_argument("--model_path", type=str, 
                      help="Path to fine-tuned model (optional)")
    parser.add_argument("--source", type=str, required=True,
                      help="Source text to translate")
    parser.add_argument("--src_lang", type=str, required=True,
                      help="Source language code")
    parser.add_argument("--tgt_lang", type=str, required=True,
                      help="Target language code")
    
    args = parser.parse_args()
    
    # Get language codes for the model type
    lang_codes = get_language_codes(args.model_type)
    print(f"Using {args.model_type} model with language codes: {lang_codes}")
    
    # Validate language codes
    if args.src_lang not in lang_codes.values():
        print(f"Error: Source language code '{args.src_lang}' is not valid for {args.model_type}.")
        print(f"Valid codes are: {list(lang_codes.values())}")
        return
    
    if args.tgt_lang not in lang_codes.values():
        print(f"Error: Target language code '{args.tgt_lang}' is not valid for {args.model_type}.")
        print(f"Valid codes are: {list(lang_codes.values())}")
        return
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer...")
    peft_model_path = args.model_path if args.model_path else None
    model, tokenizer = get_model_and_tokenizer(args.model_type, peft_model_path=peft_model_path)
    
    # Translate text
    print(f"Translating from {args.src_lang} to {args.tgt_lang}...")
    translation = translate(
        model,
        tokenizer,
        args.source,
        args.src_lang,
        args.tgt_lang
    )
    
    # Print results
    print("\nTranslation Results:")
    print(f"Source ({args.src_lang}): {args.source}")
    print(f"Translation ({args.tgt_lang}): {translation}")


if __name__ == "__main__":
    main()
