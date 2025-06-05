#!/bin/bash
# Train on Apple M1/M2 (MPS) with Accelerate
# The --auto_config flag will automatically set appropriate values for MPS:
# - Disables fp16 and bf16
# - Sets appropriate batch sizes

# Exit on error
set -e

# Define parameters for easier reuse
MODEL_TYPE="nllb"
SRC_LANG="en"
TGT_LANG="ff"
DATA_PATH="data/adamawa_english_fulfulde.jsonl"
OUTPUT_DIR="models/nllb_en_ff"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start training
echo "Starting training with $MODEL_TYPE model for $SRC_LANG → $TGT_LANG translation..."

# Run with explicit accelerate parameters for MPS
echo "Using accelerate with explicit MPS settings..."
python -m accelerate.commands.launch \
  --mixed_precision=no \
  --num_processes=1 \
  --num_machines=1 \
  --dynamo_backend=no \
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
    --max_samples 50

  echo "Evaluation complete. Results are shown above."
  echo "You can also monitor with: tensorboard --logdir $OUTPUT_DIR"
else
  echo "Training failed or no model was saved. Check the logs above for errors."
  exit 1
fi
