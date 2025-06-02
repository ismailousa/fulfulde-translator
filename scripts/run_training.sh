#!/bin/bash
# Shell script to launch training for specific translation directions

# Default values
MODEL_TYPE="nllb"
DATA_PATH="../data/adamawa_english_fulfulde_french_fub.jsonl"
OUTPUT_DIR="../models"
USE_PEFT=true
NUM_EPOCHS=3
BATCH_SIZE=16

# Parse command-line options
while getopts "m:s:t:d:o:p:e:b:h" opt; do
  case $opt in
    m) MODEL_TYPE=$OPTARG ;;
    s) SRC_LANG=$OPTARG ;;
    t) TGT_LANG=$OPTARG ;;
    d) DATA_PATH=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    p) USE_PEFT=$OPTARG ;;
    e) NUM_EPOCHS=$OPTARG ;;
    b) BATCH_SIZE=$OPTARG ;;
    h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -m MODEL_TYPE    Model type (nllb or m2m100) [default: nllb]"
      echo "  -s SRC_LANG      Source language code"
      echo "  -t TGT_LANG      Target language code"
      echo "  -d DATA_PATH     Path to the JSONL data file [default: ../data/adamawa_english_fulfulde_french_fub.jsonl]"
      echo "  -o OUTPUT_DIR    Directory to save the model [default: models/<model>_<src>_to_<tgt>_<peft|dense>_<datetime>]"
      echo "  -p USE_PEFT      Whether to use PEFT/LoRA [default: true]"
      echo "  -e NUM_EPOCHS    Number of training epochs [default: 3]"
      echo "  -b BATCH_SIZE    Batch size for training [default: 16]"
      echo "  -h               Display this help message"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Check required arguments
if [ -z "$SRC_LANG" ] || [ -z "$TGT_LANG" ]; then
  echo "Error: Source language (-s) and target language (-t) must be specified."
  echo "Run '$0 -h' for usage information."
  exit 1
fi

# If OUTPUT_DIR is not set, generate a descriptive, self-generated directory name
if [ -z "$OUTPUT_DIR" ]; then
  DATE_STR=$(date +%Y%m%d_%H%M%S)
  if [ "$USE_PEFT" = true ] || [ "$USE_PEFT" = "true" ] || [ "$USE_PEFT" = 1 ]; then
    PEFT_TAG="peft"
  else
    PEFT_TAG="dense"
  fi
  OUTPUT_DIR="models/${MODEL_TYPE}_${SRC_LANG}_to_${TGT_LANG}_${PEFT_TAG}_${DATE_STR}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== Training Configuration ==="
echo "Model type:       $MODEL_TYPE"
echo "Source language:  $SRC_LANG"
echo "Target language:  $TGT_LANG"
echo "Data path:        $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Use PEFT/LoRA:    $USE_PEFT"
echo "Number of epochs: $NUM_EPOCHS"
echo "Batch size:       $BATCH_SIZE"
echo "==========================="

# Run training script
poetry run python -m src.training.train \
  --model_type "$MODEL_TYPE" \
  --src_lang "$SRC_LANG" \
  --tgt_lang "$TGT_LANG" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs "$NUM_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  ${USE_PEFT:+--use_peft}

# Exit status
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
  echo "Model saved to: $OUTPUT_DIR"
else
  echo "Training failed with exit code $?."
fi
