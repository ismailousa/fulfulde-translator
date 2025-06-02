#!/usr/bin/env python
"""
Download the stephanedonna/english_fulfulde dataset from Hugging Face
and save it as JSONL format for Fulfulde-English translation.
"""
import os
import json
import argparse
from datasets import load_dataset

def download_and_save_dataset(output_file):
    """
    Download the dataset and save it as JSONL.
    
    Args:
        output_file: Path to the output JSONL file
    """
    print(f"Downloading 'stephanedonna/english_fulfulde' dataset...")
    
    # Load the dataset
    ds = load_dataset("stephanedonna/english_fulfulde")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as JSONL
    print(f"Dataset loaded. Contains {len(ds['train'])} training examples.")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for i, example in enumerate(ds['train']):
            # Create JSON object with language codes as keys
            json_obj = {
                "english": example['source'],
                "fulfulde": example['target']
            }
            
            # Write to JSONL file
            jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            
            # Print progress
            if (i + 1) % 1000 == 0 or i + 1 == len(ds['train']):
                print(f"Processed {i+1}/{len(ds['train'])} examples", end='\r')
    
    print(f"\nConversion complete. Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Download and convert English-Fulfulde dataset to JSONL format.')
    parser.add_argument('--output-file', type=str, default='data/adamawa_linjila_english_fulfulde.jsonl', 
                        help='Path to the output JSONL file')
    
    args = parser.parse_args()
    download_and_save_dataset(args.output_file)

if __name__ == "__main__":
    main()
