
"""
Tools for distilling trained models into smaller, more efficient versions.
"""
import os
import argparse
from typing import Optional, Dict, Any
import yaml
import torch
import ctranslate2
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel

# Add the project root to path for absolute imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules with absolute imports
from src.data.dataset_utils import process_dataset_for_model
from src.training.train import preprocess_function, compute_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def export_to_ctranslate2(
    model_path: str,
    output_dir: str,
    quantization: str = "int8",
    device: str = "cpu"
):
    """
    Export a Hugging Face model to CTranslate2 format.
    
    Args:
        model_path: Path to the Hugging Face model
        output_dir: Output directory for CTranslate2 model
        quantization: Quantization type (int8, int16, float16)
        device: Device to optimize for (cpu or cuda)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model to CTranslate2 format
    ctranslate2.converters.convert_from_pretrained(
        model_path,
        output_dir=output_dir,
        quantization=quantization,
        device=device
    )
    
    print(f"Model exported to CTranslate2 format at: {output_dir}")


def export_to_onnx(
    model_path: str,
    output_dir: str,
    opset_version: int = 12
):
    """
    Export a Hugging Face model to ONNX format.
    
    Args:
        model_path: Path to the Hugging Face model
        output_dir: Output directory for ONNX model
        opset_version: ONNX opset version
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Example input for tracing
    inputs = tokenizer("This is a test", return_tensors="pt")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs.input_ids, inputs.attention_mask),
        f"{output_dir}/model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=opset_version
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model exported to ONNX format at: {output_dir}")


def knowledge_distillation(
    teacher_model_path: str,
    student_model_name: str,
    data_path: str,
    output_dir: str,
    src_lang: str,
    tgt_lang: str,
    model_type: str = "nllb",
    config_path: Optional[str] = None,
    is_peft_model: bool = True
):
    """
    Perform knowledge distillation from a teacher model to a student model.
    
    Args:
        teacher_model_path: Path to the teacher model
        student_model_name: Name of the student model (or path)
        data_path: Path to the data file
        output_dir: Output directory for the distilled model
        src_lang: Source language code
        tgt_lang: Target language code
        model_type: Model type (nllb or m2m100)
        config_path: Path to the configuration file
        is_peft_model: Whether the teacher model is a PEFT model
    """
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
    
    config = load_config(config_path)
    
    # Load teacher model and tokenizer
    if is_peft_model:
        if model_type == "nllb":
            base_model_name = config["models"]["nllb"]["model_name"]
        else:
            base_model_name = config["models"]["m2m100"]["model_name"]
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        teacher_model = PeftModel.from_pretrained(base_model, teacher_model_path)
    else:
        teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_path)
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_path if not is_peft_model else base_model_name
    )
    
    # Load student model and tokenizer
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name)
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    # Prepare dataset
    dataset = process_dataset_for_model(
        data_path,
        model_type,
        src_lang,
        tgt_lang,
        test_size=config["data"]["train_test_split"],
        seed=config["data"]["seed"]
    )
    
    # Tokenize dataset with student tokenizer
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(
            examples,
            student_tokenizer,
            config["data"]["max_length"],
            src_lang,
            tgt_lang
        ),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=student_tokenizer,
        model=student_model,
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
        report_to="wandb"
    )
    
    # TODO: Implement distillation loss and training loop
    # This is a simplified version using standard training
    # For full distillation, we would need to implement custom loss
    # that combines cross-entropy with KL divergence from teacher
    
    trainer = Seq2SeqTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=student_tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, student_tokenizer, metric)
    )
    
    # Train student model
    trainer.train()
    
    # Save distilled model
    trainer.save_model(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    
    print(f"Distilled model saved at: {output_dir}")
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model distillation and export")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Knowledge distillation parser
    distill_parser = subparsers.add_parser("distill", help="Perform knowledge distillation")
    distill_parser.add_argument("--teacher_model_path", type=str, required=True,
                             help="Path to the teacher model")
    distill_parser.add_argument("--student_model_name", type=str, required=True,
                             help="Name or path of the student model")
    distill_parser.add_argument("--data_path", type=str, required=True,
                             help="Path to the data file")
    distill_parser.add_argument("--output_dir", type=str, required=True,
                             help="Output directory for the distilled model")
    distill_parser.add_argument("--src_lang", type=str, required=True,
                             help="Source language code")
    distill_parser.add_argument("--tgt_lang", type=str, required=True,
                             help="Target language code")
    distill_parser.add_argument("--model_type", type=str, default="nllb",
                             choices=["nllb", "m2m100"], help="Model type")
    distill_parser.add_argument("--config_path", type=str,
                             help="Path to the configuration file")
    distill_parser.add_argument("--is_peft_model", action="store_true",
                             help="Whether the teacher model is a PEFT model")
    
    # CTranslate2 export parser
    ct2_parser = subparsers.add_parser("export_ct2", help="Export to CTranslate2 format")
    ct2_parser.add_argument("--model_path", type=str, required=True,
                          help="Path to the Hugging Face model")
    ct2_parser.add_argument("--output_dir", type=str, required=True,
                          help="Output directory for CTranslate2 model")
    ct2_parser.add_argument("--quantization", type=str, default="int8",
                          choices=["int8", "int16", "float16"], 
                          help="Quantization type")
    ct2_parser.add_argument("--device", type=str, default="cpu",
                          choices=["cpu", "cuda"], help="Device to optimize for")
    
    # ONNX export parser
    onnx_parser = subparsers.add_parser("export_onnx", help="Export to ONNX format")
    onnx_parser.add_argument("--model_path", type=str, required=True,
                           help="Path to the Hugging Face model")
    onnx_parser.add_argument("--output_dir", type=str, required=True,
                           help="Output directory for ONNX model")
    onnx_parser.add_argument("--opset_version", type=int, default=12,
                           help="ONNX opset version")
    
    args = parser.parse_args()
    
    if args.command == "distill":
        knowledge_distillation(
            args.teacher_model_path,
            args.student_model_name,
            args.data_path,
            args.output_dir,
            args.src_lang,
            args.tgt_lang,
            args.model_type,
            args.config_path,
            args.is_peft_model
        )
    elif args.command == "export_ct2":
        export_to_ctranslate2(
            args.model_path,
            args.output_dir,
            args.quantization,
            args.device
        )
    elif args.command == "export_onnx":
        export_to_onnx(
            args.model_path,
            args.output_dir,
            args.opset_version
        )
    else:
        parser.print_help()
