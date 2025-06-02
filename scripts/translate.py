#!/usr/bin/env python3
"""
Launcher script for Fulfulde-English/French translation using fine-tuned models.
This is a simple wrapper around the implementation in src/inference/translate.py.
"""
import os
import sys

# Add project root to path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.inference.translate import main

if __name__ == "__main__":
    main()
