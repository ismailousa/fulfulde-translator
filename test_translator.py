#!/usr/bin/env python3
"""
Test script for the Fulfulde translator.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_translator():
    """Test the Fulfulde translator with some example phrases."""
    print("Testing Fulfulde translator...")
    
    # Import the translator
    from model.inference import load_translator
    
    # Initialize the translator
    print("\nInitializing translator...")
    translator = load_translator()
    print(f"Using device: {translator.device}")
    
    # Test phrases
    test_phrases = [
        "Mi yahay",  # I am going
        "A jaɓɓaa?",  # Do you understand?
        "Miɗo yahay suudu",  # I am going home
        "Ɗum saare Giɗaaɗo naa?",  # Is that Giɗaaɗo's house?
        "Ɓe ɗon jooɗi ton."  # They are sitting there.
    ]
    
    # Translate each phrase
    print("\nTesting translations:")
    print("-" * 50)
    for phrase in test_phrases:
        try:
            translation = translator.translate(phrase)
            print(f"Fulfulde: {phrase}")
            print(f"English:  {translation}")
            print("-" * 50)
        except Exception as e:
            print(f"Error translating '{phrase}': {str(e)}")
    
    # Test batch translation
    print("\nTesting batch translation:")
    translations = translator.translate(test_phrases)
    for fulfulde, english in zip(test_phrases, translations):
        print(f"{fulfulde} -> {english}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_translator()
