#!/bin/bash
# Training on RunPod (CUDA GPU)
# 1. Clone your repo or upload via RunPod UI
# 2. Install requirements
# 3. Launch training with accelerate

# Exit on error
set -e

# Define parameters for easier reuse
MODEL_TYPE="nllb"
SRC_LANG="ff"
TGT_LANG="en"
DATA_PATH="data/adamawa_english_fulfulde.jsonl"
OUTPUT_DIR="outputs/nllb_ff_en"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# These commands should be run once when setting up the pod
# git clone https://github.com/YOU/fulfulde-nllb
# cd fulfulde-nllb
# pip install -r requirements.txt

# Start training
echo "Starting training with $MODEL_TYPE model for $SRC_LANG → $TGT_LANG translation..."

# Run with accelerate for CUDA
python -m accelerate.commands.launch \
  --mixed_precision=fp16 \
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

# Check if training was successful (if the model folder has files)
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
  echo "Training complete. Running evaluation..."
  python src/evaluation/evaluate_model.py \
    --model_dir $OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --src_lang $SRC_LANG \
    --tgt_lang $TGT_LANG \
    --data_path $DATA_PATH \
    --use_peft \
    --max_samples 100

  echo "Evaluation complete. Results are shown above."
else
  echo "Training failed or no model was saved. Check the logs above for errors."
  exit 1
fi
