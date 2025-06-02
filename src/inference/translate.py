"""
Translation script implementation for Fulfulde-English/French translation.
"""
import os
import sys
import argparse
from typing import Optional, Dict, Any

# Add project root to path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.inference.translator import Translator, get_language_code


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate text using fine-tuned NLLB or M2M100 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model options
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True,
        help="Path to the model directory containing the fine-tuned model"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["nllb", "m2m100"], 
        default="nllb",
        help="Model architecture type"
    )
    parser.add_argument(
        "--use_peft", 
        action="store_true", 
        help="Whether the model was trained with PEFT/LoRA"
    )
    
    # Device options
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "mps", "cpu"], 
        default=None,
        help="Device to run inference on (auto-detect if not specified)"
    )
    
    # Language options
    parser.add_argument(
        "--src_lang", 
        type=str, 
        required=True,
        help="Source language (e.g., 'ff', 'en', 'fr' or 'fulfulde', 'english', 'french')"
    )
    parser.add_argument(
        "--tgt_lang", 
        type=str, 
        required=True,
        help="Target language (e.g., 'ff', 'en', 'fr' or 'fulfulde', 'english', 'french')"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", 
        type=str,
        help="Text to translate"
    )
    input_group.add_argument(
        "--input_file", 
        type=str,
        help="Input file with text to translate (one text per line, or JSON/JSONL)"
    )
    parser.add_argument(
        "--input_key", 
        type=str,
        help="For JSON files, the key to read input text from"
    )
    
    # Output options
    parser.add_argument(
        "--output_file", 
        type=str,
        help="Output file to write translations to (required if input_file is specified)"
    )
    parser.add_argument(
        "--output_key", 
        type=str,
        help="For JSON files, the key to write translations to"
    )
    
    # Generation options
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=128,
        help="Maximum length of generated translation"
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=5,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for processing multiple inputs at once"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Validate arguments
    if parsed_args.input_file and not parsed_args.output_file:
        parser.error("--output_file is required when --input_file is specified")
    
    return parsed_args


def run_translation(args: Optional[Dict[str, Any]] = None):
    """
    Run translation with the given arguments.
    
    Args:
        args: Dictionary of arguments. If None, parse from command line.
    """
    # Parse arguments
    if args is None:
        args = vars(parse_args())
    
    # Convert language names to model-specific codes
    src_lang = get_language_code(args["src_lang"], args["model_type"])
    tgt_lang = get_language_code(args["tgt_lang"], args["model_type"])
    
    print(f"Initializing translator for {args['model_type']} model...")
    print(f"Translation direction: {src_lang} → {tgt_lang}")
    
    # Initialize translator
    translator = Translator(
        model_dir=args["model_dir"],
        model_type=args["model_type"],
        use_peft=args.get("use_peft", False),
        device=args.get("device"),
        batch_size=args.get("batch_size", 8),
    )
    
    # Translate text or file
    if "text" in args and args["text"]:
        translation = translator.translate(
            text=args["text"],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=args.get("max_length", 128),
            num_beams=args.get("num_beams", 5),
        )
        print(f"\nSource ({src_lang}): {args['text']}")
        print(f"Translation ({tgt_lang}): {translation}")
        return translation
    else:
        translator.translate_file(
            input_file=args["input_file"],
            output_file=args["output_file"],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            input_key=args.get("input_key"),
            output_key=args.get("output_key"),
            max_length=args.get("max_length", 128),
            num_beams=args.get("num_beams", 5),
        )
        return None


def main():
    """Main entry point for the translation script."""
    run_translation()


if __name__ == "__main__":
    main()
