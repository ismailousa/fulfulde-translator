#!/usr/bin/env python
"""
Convert the health corpus from TSV to JSONL format.
Specifically for converting corps_sante_english_fulfulde_french_fub.tsv to JSONL.
"""
import os
import json
import csv

# Input and output file paths
input_file = "data/corps_sante_english_fulfulde_french_fub.tsv"
output_file = "data/adamawa_health_english_fulfulde_french.jsonl"

# Language codes for the three columns
languages = ["en", "ff", "fr"]

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"Converting health corpus from {input_file} to {output_file}...")

# Process the file
with open(input_file, 'r', encoding='utf-8') as tsv_file, open(output_file, 'w', encoding='utf-8') as jsonl_file:
    # Use CSV reader to properly handle TSV
    reader = csv.reader(tsv_file, delimiter='\t')
    count = 0
    
    for row in reader:
        # Skip empty rows or rows with wrong number of columns
        if not row or len(row) != 3:
            continue
            
        # Create an object with language codes as keys
        entry = {
            languages[0]: row[0].strip(),
            languages[1]: row[1].strip(),
            languages[2]: row[2].strip()
        }
        
        # Write as JSON line
        jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        count += 1
        
print(f"Conversion complete! {count} entries written to {output_file}")
print(f"You can now use this file for training or evaluation.")
