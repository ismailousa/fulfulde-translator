#!/usr/bin/env python
"""
Convert a TSV file with multiple languages to JSONL format.
Each line in JSONL will be a JSON object with language codes as keys.
"""
import os
import json
import argparse
import csv
from pathlib import Path

def convert_tsv_to_jsonl(input_file, output_file, lang_codes=None):
    """
    Convert a TSV file to JSONL format.
    
    Args:
        input_file: Path to the TSV file
        output_file: Path to the output JSONL file
        lang_codes: List of language codes corresponding to columns (default: english,fulfulde,french)
    """
    if lang_codes is None:
        lang_codes = ["english", "fulfulde", "french"]
    
    print(f"Converting {input_file} to {output_file}")
    print(f"Using language codes: {lang_codes}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Count the lines
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    # Convert TSV to JSONL
    with open(input_file, 'r', encoding='utf-8') as tsv_file:
        # Using csv reader to handle potential quoting in TSV
        reader = csv.reader(tsv_file, delimiter='\t')
        
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for i, row in enumerate(reader):
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                # Ensure we have enough columns
                if len(row) != len(lang_codes):
                    print(f"Warning: Line {i+1} has {len(row)} columns, expected {len(lang_codes)}. Skipping.")
                    continue
                
                # Create JSON object
                json_obj = {lang: text.strip() for lang, text in zip(lang_codes, row)}
                
                # Write to JSONL file
                jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                
                # Print progress
                if (i + 1) % 100 == 0 or i + 1 == line_count:
                    print(f"Processed {i+1}/{line_count} lines", end='\r')
    
    print(f"\nConversion complete. Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert TSV file to JSONL format.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file')
    parser.add_argument('--output-file', type=str, help='Path to the output JSONL file. If not provided, will use the input filename with .jsonl extension.')
    parser.add_argument('--lang-codes', type=str, default='en,ff,fr', help='Comma-separated language codes for columns (default: en,ff,fr)')
    
    args = parser.parse_args()
    
    # If output file is not provided, use input filename with .jsonl extension
    if not args.output_file:
        input_path = Path(args.input_file)
        output_file = str(input_path.with_suffix('.jsonl'))
    else:
        output_file = args.output_file
    
    # Parse language codes
    lang_codes = args.lang_codes.split(',')
    
    convert_tsv_to_jsonl(args.input_file, output_file, lang_codes)

if __name__ == "__main__":
    main()
