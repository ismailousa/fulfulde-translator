#!/bin/bash
# Training in Hugging Face Spaces

# Exit on error
set -e

# Define parameters for easier reuse
MODEL_TYPE="nllb"
SRC_LANG="ff"
TGT_LANG="fr"
DATA_PATH="data/adamawa_english_fulfulde_french_fub.jsonl"
OUTPUT_DIR="outputs/nllb_ff_fr"

# =================================================================
# Option A: No-code AutoTrain (see https://huggingface.co/autotrain)
# =================================================================
# 1. Push your dataset to the Hub
# 2. Go to https://huggingface.co/autotrain
# 3. Upload dataset, configure ff ↔ en/fr, select NLLB + LoRA

# =================================================================
# Option B: Code-based training in Space (use this in app.py)
# =================================================================

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start training
echo "Starting training with $MODEL_TYPE model for $SRC_LANG → $TGT_LANG translation..."

# Using python -m to ensure proper module imports
python3 -m accelerate.commands.launch \
  --mixed_precision=no \
  --num_processes=1 \
  --num_machines=1 \
  src/training/train.py \
  --auto_config \
  --model_type $MODEL_TYPE \
  --src_lang $SRC_LANG \
  --tgt_lang $TGT_LANG \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --use_peft

# Check if training was successful
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
  echo "Training complete. Running evaluation..."
  python3 src/evaluation/evaluate_model.py \
    --model_dir $OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --src_lang $SRC_LANG \
    --tgt_lang $TGT_LANG \
    --data_path $DATA_PATH \
    --use_peft \
    --max_samples 50

  echo "Evaluation complete. Results are shown above."
else
  echo "Training failed or no model was saved. Check the logs above for errors."
  exit 1
fi

# Note: Free Spaces = small CPUs only, good for short experiments/hosting.
