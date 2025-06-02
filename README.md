# Fulfulde Translator

> Fine-tuning NLLB & M2M100 models for Fulfulde ↔ English/French translation

## 🌍 Overview

This project focuses on fine-tuning multilingual translation models to improve translation quality between Fulfulde (ff), English (en), and French (fr), with a particular focus on the Fulfulde dialect. The implementation uses Parameter-Efficient Fine-Tuning (PEFT/LoRA) for training efficiency and provides tools for model distillation to create smaller, deployment-ready models.

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.6.0+
- Transformers 4.48.3+
- PEFT 0.7.1+

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fulfulde-translator.git
   cd fulfulde-translator
   ```

2. Install dependencies:
   ```bash
   # Using pip
   pip install -e .
   
   # Or using Poetry
   poetry install
   ```

## 🧠 Project Structure

```
fulfulde-translator/
├── data/                           # Training and evaluation data
│   └── adamawa_english_fulfulde_french_fub.jsonl
├── src/                            # Source code
│   ├── config/                     # Configuration files
│   ├── models/                     # Model utilities
│   ├── tokenization/               # Tokenizer utilities
│   ├── data/                       # Dataset processing
│   ├── training/                   # Training code
│   ├── evaluation/                 # Evaluation metrics
│   └── distillation/               # Model distillation 
├── scripts/                        # Shell scripts for training
├── examples/                       # Example usage scripts
├── tests/                          # Unit tests
└── README.md                       # This file
```

## 📊 Data Format

The training data is provided in `.jsonl` format with the following structure:

```json
{
  "english": "Buuba is Fulani.",
  "fulfulde": "Buuba pullo.",
  "french": "Buuba est Peul."
}
```

The code automatically converts this to the Hugging Face translation format:

```json
{
  "translation": {
    "ff": "Buuba pullo.",
    "en": "Buuba is Fulani.",
    "fr": "Buuba est Peul."
  }
}
```

## 📝 Usage

### Training a Model

Use the provided training script to fine-tune a model for a specific translation direction:

```bash
# From the project root directory
chmod +x scripts/run_training.sh
./scripts/run_training.sh -m nllb -s en -t ff -d data/adamawa_english_fulfulde_french_fub.jsonl -o models
```

Parameters:
- `-m`: Model type (`nllb` or `m2m100`)
- `-s`: Source language code
- `-t`: Target language code
- `-d`: Path to data file
- `-o`: Output directory
- `-p`: Whether to use PEFT/LoRA (default: true)
- `-e`: Number of training epochs
- `-b`: Batch size

### Language Codes

| Language | NLLB Code | M2M100 Code |
|----------|-----------|-------------|
| Fulfulde | `ff`      | `ff_Latn`   |
| English  | `en`      | `en_Latn`   |
| French   | `fr`      | `fr_Latn`   |

### Running Translation

Use the example script to translate text:

```bash
python examples/translate_example.py --model_type nllb --src_lang en --tgt_lang ff --source "Hello, how are you?"
```

For a fine-tuned model:

```bash
python examples/translate_example.py --model_type nllb --model_path ./models/nllb_en_to_ff --src_lang en --tgt_lang ff --source "Hello, how are you?"
```

### Evaluating Model Performance

To evaluate model performance:

```python
from src.evaluation.eval_metrics import run_full_evaluation

results = run_full_evaluation(
    model_path="./models/nllb_en_to_ff",
    test_data_path="./data/test_data.json",
    src_lang="en",
    tgt_lang="ff",
    model_type="nllb",
    output_dir="./evaluation_results",
    is_peft_model=True
)
```

### Model Distillation

To distill a fine-tuned model into a smaller one:

```python
from src.distillation.distill import knowledge_distillation

knowledge_distillation(
    teacher_model_path="./models/nllb_en_to_ff",
    student_model_name="facebook/nllb-200-distilled-100M",
    data_path="./data/adamawa_english_fulfulde_french_fub.jsonl",
    output_dir="./distilled_models/nllb_en_to_ff_small",
    src_lang="en",
    tgt_lang="ff",
    model_type="nllb",
    is_peft_model=True
)
```

To export a model for inference:

```python
from src.distillation.distill import export_to_ctranslate2

export_to_ctranslate2(
    model_path="./distilled_models/nllb_en_to_ff_small",
    output_dir="./deployment_models/nllb_en_to_ff_ct2",
    quantization="int8",
    device="cpu"
)
```

## 🧪 Running Tests

Run the test suite to ensure everything is working correctly:

```bash
python -m pytest tests/
```

## 📈 Evaluation Metrics

The project uses the following metrics for evaluation:

- **SacreBLEU**: Standard metric for translation quality
- **Round-trip BLEU**: Consistency check by translating back to the original language
- **Error Analysis**: Detailed examination of the worst-performing translations

## 🔮 Future Work

- Support for additional Fulfulde dialects
- Web interface for translation
- Mobile deployment with optimized models
- Integration with speech recognition for spoken Fulfulde translation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers and PEFT libraries
- Meta AI for the NLLB and M2M100 models
- Contributors to the Fulfulde linguistic resources
