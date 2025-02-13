import pdfplumber
import os
import argparse
import pandas as pd
from typing import List

def extract_text_from_pdf(pdf_path: str, output_dir: str, verbose=False) -> None:
    """Extracts text from a PDF file.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory where extracted text will be saved.
        verbose (bool): If True, prints progress updates.
    """
    # Check if file exists
    if not os.path.isfile(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    output_filename = os.path.basename(pdf_path)
    output_filename_no_ext = os.path.splitext(output_filename)[0]
    
    if verbose:
        print(f"Extracted text from '{pdf_path}' and saved it to '{output_dir}/{output_filename_no_ext}.txt'")

    save_path = f"{output_dir}/{output_filename_no_ext}.txt"
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(text)

def main():
    parser = argparse.ArgumentParser(description='Extract languages from a PDF file')
    parser.add_argument('pdf_file', help='Path to the input PDF file')
    parser.add_argument('-o', '--output_dir', default='./data/extracted/', help='Output directory for extracted files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist. Creating it...")
        os.makedirs(args.output_dir)

    extract_text_from_pdf(args.pdf_file, args.output_dir, args.verbose)

if __name__ == "__main__":
    main()