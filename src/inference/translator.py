"""
Inference module for running translation with fine-tuned models.
"""
import os
import torch
from typing import List, Dict, Any, Optional, Union
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Add project root to path for absolute imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Translator:
    """
    Translator class for handling translations between Fulfulde and other languages.
    Supports both NLLB and M2M100 models, with or without PEFT/LoRA.
    """
    
    def __init__(
        self,
        model_dir: str,
        model_type: str = "nllb",
        use_peft: bool = True,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        """
        Initialize the translator with the fine-tuned model.
        
        Args:
            model_dir: Path to the model directory (with adapter_config.json for PEFT)
            model_type: Either "nllb" or "m2m100"
            use_peft: Whether the model was trained with PEFT/LoRA
            device: Device to run inference on (cuda, mps, cpu). If None, will auto-detect
            batch_size: Batch size for processing multiple inputs at once
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.use_peft = use_peft
        self.batch_size = batch_size
        
        # Detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer based on configuration."""
        # Determine base model ID
        if self.model_type == "nllb":
            base_model_id = "facebook/nllb-200-distilled-600M"
        else:  # m2m100
            base_model_id = "facebook/m2m100_418M"
        
        # Load model
        if self.use_peft:
            print(f"Loading PEFT model from {self.model_dir} with base {base_model_id}")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        else:
            print(f"Loading full model from {self.model_dir}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")) 
            else base_model_id
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def translate(
        self, 
        text: Union[str, List[str]], 
        src_lang: str, 
        tgt_lang: str, 
        max_length: int = 128,
        num_beams: int = 5,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Input text or list of texts to translate
            src_lang: Source language code (e.g., "ff" for NLLB, "ff_Latn" for M2M100)
            tgt_lang: Target language code (e.g., "en" for NLLB, "en_Latn" for M2M100)
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search
            **kwargs: Additional generation parameters
            
        Returns:
            Translated text or list of translated texts
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        # Set source and target languages
        self.tokenizer.src_lang = src_lang
        
        # For NLLB models, we need to handle forced BOS token differently
        if self.model_type == "nllb":
            # For NLLB, forced BOS token is in format "__<lang>__"
            forced_bos_token = f"__{tgt_lang}__"
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(forced_bos_token)
        else:  # M2M100
            # Set target language for M2M100
            self.tokenizer.tgt_lang = tgt_lang
            # M2M100 has lang_code_to_id attribute
            forced_bos_token_id = self.tokenizer.get_lang_id(tgt_lang)
        
        results = []
        # Process in batches to avoid OOM errors
        for i in range(0, len(text), self.batch_size):
            batch = text[i:i+self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translations
            generate_kwargs = {
                **inputs,
                "forced_bos_token_id": forced_bos_token_id,
                "max_length": max_length,
                "num_beams": num_beams,
                **kwargs
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)
            
            # Decode outputs
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(translations)
        
        # Return single result if input was single
        if is_single:
            return results[0]
        return results
    
    def translate_file(
        self,
        input_file: str,
        output_file: str,
        src_lang: str,
        tgt_lang: str,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Translate a file with one text per line.
        If the file is JSON or JSONL, specify keys to read from/write to.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            src_lang: Source language code
            tgt_lang: Target language code
            input_key: For JSON(L) files, the key to read input text from
            output_key: For JSON(L) files, the key to write translations to
            **kwargs: Additional translation parameters
        """
        input_texts = []
        is_json = False
        is_jsonl = False
        json_objects = []
        
        # Determine file type and read content
        if input_file.endswith(('.json', '.jsonl')):
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    # Try to read as single JSON
                    content = f.read().strip()
                    if not content:
                        print(f"Warning: Empty file {input_file}")
                        return
                    
                    # Check if it's JSONL (multiple JSON objects, one per line)
                    if '\n' in content and input_file.endswith('.jsonl'):
                        is_jsonl = True
                        for line in content.split('\n'):
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            json_objects.append(obj)
                            if input_key:
                                input_texts.append(obj.get(input_key, ""))
                            else:
                                # If no key specified, use the whole object as string
                                input_texts.append(str(obj))
                    else:
                        # Single JSON object
                        is_json = True
                        obj = json.loads(content)
                        json_objects.append(obj)
                        if input_key:
                            if isinstance(obj.get(input_key), list):
                                input_texts.extend(obj.get(input_key, []))
                            else:
                                input_texts.append(obj.get(input_key, ""))
                        else:
                            # If no key specified, use the whole object as string
                            input_texts.append(str(obj))
                except json.JSONDecodeError:
                    # Not valid JSON, treat as text file
                    pass
        
        # If not JSON/JSONL or couldn't parse, treat as text file
        if not is_json and not is_jsonl:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        input_texts.append(line)
        
        # Translate all texts
        if not input_texts:
            print(f"Warning: No texts found in {input_file}")
            return
            
        translations = self.translate(input_texts, src_lang, tgt_lang, **kwargs)
        
        # Write results
        with open(output_file, 'w', encoding='utf-8') as f:
            if is_jsonl:
                for i, (obj, translation) in enumerate(zip(json_objects, translations)):
                    if output_key:
                        obj[output_key] = translation
                    else:
                        obj['translation'] = translation
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            elif is_json:
                obj = json_objects[0]
                if output_key:
                    if isinstance(obj.get(input_key), list):
                        obj[output_key] = translations
                    else:
                        obj[output_key] = translations[0]
                else:
                    obj['translation'] = translations[0] if len(translations) == 1 else translations
                f.write(json.dumps(obj, ensure_ascii=False, indent=2))
            else:
                # Plain text
                for translation in translations:
                    f.write(translation + '\n')
        
        print(f"Translated {len(input_texts)} texts from {src_lang} to {tgt_lang}")
        print(f"Results saved to {output_file}")


def get_language_code(language: str, model_type: str) -> str:
    """
    Convert language name to the correct code for the model type.
    
    Args:
        language: Language name or code
        model_type: Model type (nllb or m2m100)
        
    Returns:
        Language code for the specified model
    """
    # Lowercase for case-insensitive matching
    language = language.lower()
    
    # Common language mappings
    language_map = {
        # English variants
        "english": "en" if model_type == "nllb" else "en_Latn",
        "en": "en" if model_type == "nllb" else "en_Latn",
        
        # French variants
        "french": "fr" if model_type == "nllb" else "fr_Latn",
        "fr": "fr" if model_type == "nllb" else "fr_Latn",
        
        # Fulfulde variants
        "fulfulde": "ff" if model_type == "nllb" else "ff_Latn",
        "fula": "ff" if model_type == "nllb" else "ff_Latn",
        "fulani": "ff" if model_type == "nllb" else "ff_Latn",
        "ff": "ff" if model_type == "nllb" else "ff_Latn",
    }
    
    # Return the mapped code or the original if not found
    return language_map.get(language, language)
