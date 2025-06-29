#!/bin/bash
# Convenient wrapper script for translation using trained models

# Exit on error
set -e

# Default values
MODEL_DIR="models/nllb_en_ff_2806"
# MODEL_DIR="outputs/nllb_ff_en_pod"
MODEL_TYPE="nllb"
USE_PEFT="--use_peft"
SRC_LANG="en"
TGT_LANG="ff"

# Display usage information
function show_usage {
  echo "Fulfulde Translator - Translate between Fulfulde and English/French"
  echo ""
  echo "Usage: $0 [options] \"text to translate\""
  echo ""
  echo "Options:"
  echo "  --model-dir DIR   Model directory (default: $MODEL_DIR)"
  echo "  --model-type TYPE Model type: nllb or m2m100 (default: $MODEL_TYPE)"
  echo "  --no-peft         Use full model instead of PEFT/LoRA model"
  echo "  --from LANG       Source language: ff, en, fr (default: $SRC_LANG)"
  echo "  --to LANG         Target language: ff, en, fr (default: $TGT_LANG)"
  echo "  --file FILE       Input file to translate instead of text"
  echo "  --output FILE     Output file for translations (required with --file)"
  echo "  --help            Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 \"Hello, how are you?\""
  echo "  $0 --from en --to ff \"Hello, how are you?\""
  echo "  $0 --file input.txt --output translations.txt"
  echo "  $0 --model-dir outputs/nllb_ff_fr --to fr \"Jam wana?\""
  exit 1
}

# Process command-line arguments
INPUT_FILE=""
OUTPUT_FILE=""
TEXT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --no-peft)
      USE_PEFT=""
      shift
      ;;
    --from)
      SRC_LANG="$2"
      shift 2
      ;;
    --to)
      TGT_LANG="$2"
      shift 2
      ;;
    --file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --help)
      show_usage
      ;;
    *)
      # Anything else is treated as text to translate
      TEXT="$1"
      shift
      ;;
  esac
done

# Validate arguments
if [ -n "$INPUT_FILE" ] && [ -z "$OUTPUT_FILE" ]; then
  echo "Error: --output is required when using --file"
  echo ""
  show_usage
fi

if [ -z "$INPUT_FILE" ] && [ -z "$TEXT" ]; then
  echo "Error: Please provide text to translate or an input file"
  echo ""
  show_usage
fi

# Construct command
CMD="python scripts/translate.py --model_dir $MODEL_DIR --model_type $MODEL_TYPE $USE_PEFT --src_lang $SRC_LANG --tgt_lang $TGT_LANG"

if [ -n "$INPUT_FILE" ]; then
  CMD="$CMD --input_file $INPUT_FILE --output_file $OUTPUT_FILE"
else
  CMD="$CMD --text \"$TEXT\""
fi

# Display command (for debugging)
echo "Running translation..."

# Run translation using the restructured implementation directly
if [ -n "$INPUT_FILE" ]; then
  python -m src.inference.translate --model_dir "$MODEL_DIR" --model_type "$MODEL_TYPE" $USE_PEFT --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"
else
  python -m src.inference.translate --model_dir "$MODEL_DIR" --model_type "$MODEL_TYPE" $USE_PEFT --src_lang "$SRC_LANG" --tgt_lang "$TGT_LANG" --text "$TEXT"
fi
