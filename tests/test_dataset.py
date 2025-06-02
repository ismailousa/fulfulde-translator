"""
Unit tests for dataset utilities.
"""
import os
import unittest
import json
import tempfile
from src.data.dataset_utils import (
    load_jsonl_data,
    convert_to_translation_format,
    create_translation_pair_dataset,
    process_dataset_for_model
)


class TestDatasetUtils(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = [
            {
                "english": "Hello, how are you?",
                "fulfulde": "Jam, no mbooɗɗaa?",
                "french": "Bonjour, comment allez-vous?"
            },
            {
                "english": "What is your name?",
                "fulfulde": "Ko inneteɗaa?",
                "french": "Comment t'appelles-tu?"
            },
            {
                "english": "I am fine.",
                "fulfulde": "Miɗo jam.",
                "french": "Je vais bien."
            }
        ]
        
        # Create temporary JSONL file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.jsonl')
        for item in self.sample_data:
            self.temp_file.write(json.dumps(item) + '\n')
        self.temp_file.close()
    
    def tearDown(self):
        # Clean up temporary file
        os.unlink(self.temp_file.name)
    
    def test_load_jsonl_data(self):
        # Test loading JSONL data
        loaded_data = load_jsonl_data(self.temp_file.name)
        self.assertEqual(len(loaded_data), 3)
        self.assertEqual(loaded_data[0]["english"], "Hello, how are you?")
        self.assertEqual(loaded_data[1]["fulfulde"], "Ko inneteɗaa?")
        self.assertEqual(loaded_data[2]["french"], "Je vais bien.")
    
    def test_convert_to_translation_format_nllb(self):
        # Test converting data to NLLB translation format
        translation_data = convert_to_translation_format(self.sample_data, "nllb")
        self.assertEqual(len(translation_data), 3)
        self.assertIn("en", translation_data[0]["translation"])
        self.assertIn("ff", translation_data[0]["translation"])
        self.assertIn("fr", translation_data[0]["translation"])
        self.assertEqual(translation_data[0]["translation"]["en"], "Hello, how are you?")
    
    def test_convert_to_translation_format_m2m100(self):
        # Test converting data to M2M100 translation format
        translation_data = convert_to_translation_format(self.sample_data, "m2m100")
        self.assertEqual(len(translation_data), 3)
        self.assertIn("en_Latn", translation_data[0]["translation"])
        self.assertIn("ff_Latn", translation_data[0]["translation"])
        self.assertIn("fr_Latn", translation_data[0]["translation"])
        self.assertEqual(translation_data[0]["translation"]["ff_Latn"], "Jam, no mbooɗɗaa?")
    
    def test_create_translation_pair_dataset(self):
        # Test creating a translation pair dataset
        translation_data = convert_to_translation_format(self.sample_data, "nllb")
        dataset = create_translation_pair_dataset(translation_data, "en", "ff", test_size=0.3, seed=42)
        
        # Check dataset structure
        self.assertIn("train", dataset)
        self.assertIn("test", dataset)
        
        # Check dataset contents
        self.assertEqual(len(dataset["train"]) + len(dataset["test"]), 3)
        self.assertEqual(dataset["train"][0]["source"], self.sample_data[0]["english"])
        self.assertEqual(dataset["train"][0]["target"], self.sample_data[0]["fulfulde"])
    
    def test_process_dataset_for_model(self):
        # This is more of an integration test, but we'll mock dependencies
        # by using the temporary file we created
        dataset = process_dataset_for_model(
            self.temp_file.name,
            "nllb",
            "en",
            "ff",
            test_size=0.3,
            seed=42
        )
        
        # Check dataset structure
        self.assertIn("train", dataset)
        self.assertIn("test", dataset)
        
        # Check that the dataset contains the expected fields
        self.assertIn("source", dataset["train"].column_names)
        self.assertIn("target", dataset["train"].column_names)
        self.assertIn("src_lang", dataset["train"].column_names)
        self.assertIn("tgt_lang", dataset["train"].column_names)


if __name__ == "__main__":
    unittest.main()
