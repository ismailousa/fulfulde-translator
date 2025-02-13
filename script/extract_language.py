import os
import argparse
import pandas as pd

def extract_language(tsv_file, languages, output_dir, verbose=False):
    """Extracts text in specified languages from a TSV file and saves each language's text in separate files.

    Args:
        tsv_file (str): Path to the input TSV file.
        languages (list): List of languages to extract.
        output_dir (str): Directory to save output files.
        verbose (bool, optional): If True, prints progress updates. Defaults to False.

    Returns:
        None
    """
    # Check if file exists
    if not os.path.isfile(tsv_file):
        print(f"Error: File '{tsv_file}' not found.")
        return

    # Read the TSV file (assuming no header)
    try:
        df = pd.read_csv(tsv_file, delimiter="\t")
    except Exception as e:
        print(f"Error reading '{tsv_file}': {e}")
        return

    # Define available columns
    available_columns = ['English', 'Fulfulde', 'French']
    
    # Ensure at least one language is provided; otherwise, default to all available
    if not languages or languages[0] == '':
        languages = available_columns
    else:
        languages = [lang.strip().capitalize() for lang in languages]

    # Check if all requested languages exist in available columns
    invalid_languages = [lang for lang in languages if lang not in available_columns]
    if invalid_languages:
        print(f"Error: Unsupported language(s) - {', '.join(invalid_languages)}")
        return

    # Assign column names to the DataFrame
    df.columns = available_columns[:len(df.columns)]  # Prevents assigning more columns than available

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the filename before the first underscore
    base_filename = os.path.splitext(os.path.basename(tsv_file))[0]  # Remove extension
    main_name = base_filename.split('_')[0] if '_' in base_filename else base_filename  # Get text before first '_'

    # Save extracted text to files
    for lang in languages:
        name = lang.lower()
        output_path = os.path.join(output_dir, f"{main_name}_{name}.txt")

        try:
            df[lang].dropna().to_csv(output_path, index=False, header=True, encoding="utf-8")

            if verbose:
                print(f"Extracted '{lang}' text into {output_path}")

        except KeyError:
            print(f"Warning: Column '{lang}' not found in the TSV file. Skipping.")

    print("Extraction completed.")



def main():
    parser = argparse.ArgumentParser(description='Extract languages from a TSV file')
    parser.add_argument('tsv_file',help='Path to the TSV file')
    parser.add_argument('-l', '--languages', default=['Fulfulde'], nargs='+', help='Languages to extract ( comma-separated )')
    parser.add_argument('-o', '--output_dir', default='./data/extracted/', help='Output directory for extracted files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()

    extract_language(args.tsv_file, args.languages, args.output_dir, args.verbose)

if __name__ == "__main__":
    main()