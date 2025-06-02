import os
import json
import argparse
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model
import wandb

# Constants for model names
MODEL_MAP = {
    'nllb': 'facebook/nllb-200-distilled-600M',
    'm2m100': 'facebook/m2m100_418M',
}

LANG_CODE_MAP = {
    'nllb': {'ff': 'ff', 'en': 'en', 'fr': 'fr'},
    'm2m100': {'ff': 'ff_Latn', 'en': 'en_Latn', 'fr': 'fr_Latn'},
}

# LoRA config
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

def load_jsonl_dataset(path, src_lang, tgt_lang, model_type):
    """Load and convert .jsonl data to Hugging Face translation format."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            translation = {}
            translation[LANG_CODE_MAP[model_type][src_lang]] = ex[src_lang]
            translation[LANG_CODE_MAP[model_type][tgt_lang]] = ex[tgt_lang]
            records.append({'translation': translation})
    return Dataset.from_list(records)

def preprocess_function(examples, tokenizer, src_lang, tgt_lang, model_type):
    src_code = LANG_CODE_MAP[model_type][src_lang]
    tgt_code = LANG_CODE_MAP[model_type][tgt_lang]
    inputs = [ex[src_code] for ex in examples['translation']]
    targets = [ex[tgt_code] for ex in examples['translation']]
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding='max_length',
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length',
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to .jsonl data file')
    parser.add_argument('--model_type', type=str, choices=['nllb', 'm2m100'], required=True)
    parser.add_argument('--src_lang', type=str, required=True)
    parser.add_argument('--tgt_lang', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--wandb_project', type=str, default='fulfulde-translator')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=f"{args.model_type}-{args.src_lang}-{args.tgt_lang}")

    # Load data
    dataset = load_jsonl_dataset(args.data, args.src_lang, args.tgt_lang, args.model_type)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = dataset['train'], dataset['test']

    # Load tokenizer and model
    model_name = MODEL_MAP[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Set language codes for tokenizer if needed
    if args.model_type == 'nllb':
        tokenizer.src_lang = LANG_CODE_MAP['nllb'][args.src_lang]
        tokenizer.tgt_lang = LANG_CODE_MAP['nllb'][args.tgt_lang]
    elif args.model_type == 'm2m100':
        tokenizer.src_lang = LANG_CODE_MAP['m2m100'][args.src_lang]
        tokenizer.tgt_lang = LANG_CODE_MAP['m2m100'][args.tgt_lang]

    # Apply LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Preprocess
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer, args.src_lang, args.tgt_lang, args.model_type), batched=True)
    eval_ds = eval_ds.map(lambda x: preprocess_function(x, tokenizer, args.src_lang, args.tgt_lang, args.model_type), batched=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.model_type}_{args.src_lang}_{args.tgt_lang}"),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True,
        logging_dir="./logs",
        report_to=["wandb"],
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main() 