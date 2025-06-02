#!/usr/bin/env python
"""
Filter a JSONL file to remove entries containing '__Error__' in any field.
"""
import os
import json
import argparse

def filter_errors(input_file, output_file=None):
    """
    Filter a JSONL file to remove entries with '__Error__' in any field.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file (default: overwrites input file)
    """
    if output_file is None:
        # Create a temporary file for the filtered content
        temp_output = input_file + '.temp'
    else:
        temp_output = output_file
    
    print(f"Filtering errors from {input_file}")
    
    # Read the file and filter entries
    filtered_entries = []
    error_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                entry = json.loads(line.strip())
                
                # Check if any field contains "__Error__"
                has_error = False
                for value in entry.values():
                    if isinstance(value, str) and "__Error__" in value:
                        has_error = True
                        error_count += 1
                        break
                
                if not has_error:
                    filtered_entries.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                error_count += 1
    
    # Write the filtered entries
    with open(temp_output, 'w', encoding='utf-8') as f:
        for entry in filtered_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # If no output file specified, replace the input file
    if output_file is None:
        os.replace(temp_output, input_file)
        final_path = input_file
    else:
        final_path = output_file
    
    print(f"Filtering complete!")
    print(f"Total entries: {total_count}")
    print(f"Entries with errors removed: {error_count}")
    print(f"Remaining entries: {len(filtered_entries)}")
    print(f"Saved to: {final_path}")

def main():
    parser = argparse.ArgumentParser(description='Filter JSONL file to remove entries with errors.')
    parser.add_argument('input_file', help='Path to the input JSONL file')
    parser.add_argument('--output-file', help='Path to the output JSONL file (default: overwrites input file)')
    parser.add_argument('--error-string', default='__Error__', help='String to filter out (default: __Error__)')
    
    args = parser.parse_args()
    filter_errors(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
