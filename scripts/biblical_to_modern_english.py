#!/usr/bin/env python3
"""
Rewrite a single biblical/archaic English line into modern English using a language model.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
# from langchain_deepseek import ChatDeepSeek  # Uncomment if using DeepSeek

# Config                
INPUT_FILE = "data/adamawa_linjila_english_fulfulde.jsonl"
OUTPUT_FILE = "data/adamawa_linjila_english_fulfulde_modern.jsonl"

# Initialize model
llm = OllamaLLM(model="llama3.2:3b", temperature=0.3, max_tokens=1000)
# llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

# Prompt template with example included
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You rewrite biblical or archaic English into modern, natural English while preserving meaning."),
    ("human", 
     "Rewrite this into clear, modern English. Treat it as one unit even if it has multiple sentences.\n\n"
     "Do not have anything else in the output other than the modern English translation."
     "Example:\n"
     "Original: And Pharaoh said unto Joseph, I have dreamed a dream, and there is none that can interpret it: "
     "and I have heard say of thee, that thou canst understand a dream to interpret it.\n"
     "Rewritten: And Pharaoh said to Joseph, I had a dream, and no one can interpret it. "
     "But I’ve heard you can understand dreams and explain them.\n\n"
     "Now modernize:\nOriginal: {text}\n\nRewritten:")
])

# Define the chain
chain = prompt_template | llm | StrOutputParser()

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    output_path = Path(OUTPUT_FILE)
    processed = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)

    with input_path.open("r", encoding="utf-8") as f:
        # Skip processed lines
        for _ in range(processed):
            next(f)
        # Count remaining lines for progress bar
        remaining_lines = sum(1 for _ in f)
        f.seek(0)
        for _ in range(processed):
            next(f)
        for i, line in enumerate(tqdm(f, total=remaining_lines, desc="Modernizing"), start=processed + 1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            archaic_text = entry.get("english", "")
            if not archaic_text.strip():    
                continue

            output = chain.invoke({"text": archaic_text})
            
            # Write to output file
            with output_path.open("a", encoding="utf-8") as out_f:
                entry = {"english": output.strip(), "fulfulde": entry.get("fulfulde", ""), "french": entry.get("french", "")}
                out_f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
