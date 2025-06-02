"""
Unit tests for training utilities.
"""
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import yaml

from src.training.train import load_config, preprocess_function


class TestTrainingUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        
        # Sample configuration
        self.test_config = {
            "models": {
                "nllb": {
                    "model_name": "facebook/nllb-200-distilled-600M",
                    "language_codes": {
                        "fulfulde": "ff",
                        "english": "en",
                        "french": "fr"
                    }
                },
                "m2m100": {
                    "model_name": "facebook/m2m100_418M",
                    "language_codes": {
                        "fulfulde": "ff_Latn",
                        "english": "en_Latn",
                        "french": "fr_Latn"
                    }
                }
            },
            "peft": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none"
            },
            "training": {
                "learning_rate": 3.0e-4,
                "num_training_epochs": 3
            }
        }
        
        # Write config to file
        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_load_config(self):
        # Test loading configuration from file
        config = load_config(self.config_path)
        
        # Check that config was loaded correctly
        self.assertEqual(config["models"]["nllb"]["model_name"], "facebook/nllb-200-distilled-600M")
        self.assertEqual(config["peft"]["r"], 8)
        self.assertEqual(config["training"]["num_training_epochs"], 3)
    
    def test_preprocess_function(self):
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 1
        
        # Mock tokenizer return values
        tokenizer.return_value = {"input_ids": [[10, 20, 30], [40, 50, 60]], "attention_mask": [[1, 1, 1], [1, 1, 1]]}
        
        # Example data
        examples = {
            "source": ["Hello", "World"],
            "target": ["Bonjour", "Monde"]
        }
        
        # Call preprocess function
        result = preprocess_function(examples, tokenizer, 128, "en", "fr")
        
        # Check tokenizer was called with the right parameters
        self.assertEqual(tokenizer.src_lang, "en")
        self.assertEqual(tokenizer.tgt_lang, "fr")
        
        # Check that the function returned the expected structure
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)


if __name__ == "__main__":
    unittest.main()
