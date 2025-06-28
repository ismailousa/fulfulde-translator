"""
Core training loop for fine-tuning translation models.
Supports various training environments (local CPU/GPU, MPS, HF Spaces, RunPod).
"""
import os
import argparse
import yaml
from typing import Dict, Any, Optional

import torch
import evaluate
from datasets import DatasetDict
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

# Add the project root to path for absolute imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules with absolute imports
from src.models.model_utils import get_model_and_tokenizer
from src.data.dataset_utils import process_dataset_for_model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess_function(examples, tokenizer, max_length, src_lang, tgt_lang):
    """Tokenize inputs and targets."""
    # Set source and target language
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    # Tokenize inputs
    inputs = [ex for ex in examples["source"]]
    targets = [ex for ex in examples["target"]]
    
    model_inputs = tokenizer(
        inputs, max_length=max_length, truncation=True, padding="max_length"
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets, max_length=max_length, truncation=True, padding="max_length"
    )
    
    # Replace padding token id with -100 so it's ignored by the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer, metric):
    """Compute BLEU score for evaluation."""
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = torch.where(
        torch.tensor(labels) == -100,
        torch.tensor(tokenizer.pad_token_id),
        torch.tensor(labels)
    )
    
    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU score
    result = metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    return {"bleu": result["score"]}


def train(
    model_type: str,
    src_lang: str,
    tgt_lang: str,
    data_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    use_peft: bool = True
):
    """
    Train a translation model.
    
    Args:
        model_type: Either "nllb" or "m2m100"
        src_lang: Source language code
        tgt_lang: Target language code
        data_path: Path to the JSONL data file
        output_dir: Directory to save the model
        config_path: Path to the configuration file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        use_peft: Whether to use PEFT/LoRA
    """
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    
    config = load_config(config_path)
    
    # Override config with provided arguments
    if num_epochs is not None:
        config["training"]["num_training_epochs"] = num_epochs
    
    if batch_size is not None:
        config["training"]["per_device_train_batch_size"] = batch_size
        config["training"]["per_device_eval_batch_size"] = batch_size
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load model and tokenizer
    peft_config = config["peft"] if use_peft else None
    model, tokenizer = get_model_and_tokenizer(model_type, peft_config)

    # Detect device - this is mostly for logging, accelerator will handle actual device placement
    if torch.cuda.is_available():
        device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print(f"Device detected: {device_type}")
    
    # Load and process dataset
    dataset = process_dataset_for_model(
        data_path,
        model_type,
        src_lang,
        tgt_lang,
        test_size=config["data"]["train_test_split"],
        seed=config["data"]["seed"]
    )

    # print dataset, size of each  split, and 10 examples each also totals
    # print("Dataset:", dataset)
    # print("Size of each split:", dataset.num_rows)
    # print("10 examples each:")
    # for split in dataset:
    #     print(dataset[split].select(range(10)))
    # print("Total examples:", dataset.num_rows)

    # # also print each line in the examples eg the src and tgt
    # for split in dataset:
    #     for i in range(10):
    #         print(dataset[split][i])

    # # reset dataset
    # dataset = dataset.reset_index()
    

    # Preprocess dataset
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(
            examples,
            tokenizer,
            config["data"]["max_length"],
            src_lang,
            tgt_lang
        ),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=config["data"]["max_length"]
    )
    
    # Load metric
    metric = evaluate.load(config["evaluation"]["metric"])
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=config["evaluation"]["eval_steps"],
        save_strategy="steps",
        save_steps=config["evaluation"]["save_steps"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        weight_decay=config["training"]["weight_decay"],
        num_train_epochs=config["training"]["num_training_epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        optim=config["training"]["optimizer"],
        lr_scheduler_type=config["training"]["scheduler"],
        warmup_steps=config["training"]["num_warmup_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        generation_max_length=config["data"]["max_length"],
        predict_with_generate=True,
        report_to="wandb",
        # Let accelerate handle device placement
        no_cuda=(device_type != "cuda")
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    with accelerator.main_process_first():
        trainer.train()
    
    # Save model
    if use_peft:
        # Unwrap model if needed when using accelerator
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    
    # Save tokenizer
    with accelerator.main_process_first():
        tokenizer.save_pretrained(output_dir)
    
    return trainer


def setup_training_config():
    """Set up training configuration from YAML file based on device."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    config = load_config(config_path)
    
    # Auto-detect device and adjust config if needed
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS-specific adjustments for Apple Silicon
        config["training"]["fp16"] = False
        config["training"]["bf16"] = False
        # Smaller batch sizes for MPS to prevent OOM
        if config["training"]["per_device_train_batch_size"] > 4:
            config["training"]["per_device_train_batch_size"] = 4
            config["training"]["per_device_eval_batch_size"] = 4
    
    # Save the updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a translation model")
    parser.add_argument("--model_type", type=str, required=True, choices=["nllb", "m2m100"],
                        help="Model type (nllb or m2m100)")
    parser.add_argument("--src_lang", type=str, required=True,
                        help="Source language code")
    parser.add_argument("--tgt_lang", type=str, required=True,
                        help="Target language code")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the JSONL data file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model")
    parser.add_argument("--config_path", type=str,
                        help="Path to the configuration file")
    parser.add_argument("--auto_config", action="store_true",
                        help="Automatically configure settings based on device")
    parser.add_argument("--num_epochs", type=int,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for training")
    parser.add_argument("--use_peft", action="store_true",
                        help="Whether to use PEFT/LoRA")
    
    args = parser.parse_args()
    
    # Auto-configure if requested
    if args.auto_config:
        config_path = setup_training_config()
    else:
        config_path = args.config_path
    
    train(
        args.model_type,
        args.src_lang,
        args.tgt_lang,
        args.data_path,
        args.output_dir,
        config_path,
        args.num_epochs,
        args.batch_size,
        args.use_peft
    )
