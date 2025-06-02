"""
Unit tests for tokenizer utilities.
"""
import unittest
from unittest.mock import patch, MagicMock
from src.tokenization.tokenizer_utils import (
    preprocess_text,
    set_language_pair,
    batch_tokenize
)


class TestTokenizerUtils(unittest.TestCase):
    def test_preprocess_text(self):
        # Test text normalization
        input_text = "Hello ,  world !  How are  you?"
        expected_output = "Hello, world! How are you?"
        self.assertEqual(preprocess_text(input_text), expected_output)
        
        # Test with different punctuation
        input_text = "Test : this ; is a  test ( with ) many [ brackets ] ."
        expected_output = "Test: this; is a test (with) many [brackets]."
        self.assertEqual(preprocess_text(input_text), expected_output)
    
    def test_set_language_pair(self):
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.src_lang = None
        tokenizer.tgt_lang = None
        
        # Set language pair
        set_language_pair(tokenizer, "en", "ff")
        
        # Check that language attributes were set
        self.assertEqual(tokenizer.src_lang, "en")
        self.assertEqual(tokenizer.tgt_lang, "ff")
    
    @patch("src.tokenization.tokenizer_utils.preprocess_text")
    def test_batch_tokenize(self, mock_preprocess_text):
        # Mock dependencies
        mock_preprocess_text.side_effect = lambda x: x  # Identity function for simplicity
        
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        
        # Test data
        texts = ["Hello", "World"]
        
        # Call function
        batch_tokenize(tokenizer, texts, "en", "ff", max_length=128)
        
        # Check that tokenizer was called with the right parameters
        tokenizer.assert_called_once()
        self.assertEqual(tokenizer.src_lang, "en")
        self.assertEqual(tokenizer.tgt_lang, "ff")


if __name__ == "__main__":
    unittest.main()
