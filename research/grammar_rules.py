import re

def apply_grammar_rules(text):
    """
    Apply Fulfulde-specific grammar rules to the input text.
    
    This function:
    1. Identifies and marks noun classes
    2. Marks verb voices (active, middle, passive)
    3. Handles consonant mutations
    """
    # Define noun class suffixes
    nc_suffixes = {
        "NC1": r'\b\w+(?:ɗo|o)\b',  # Class 1 (singular person)
        "NC2": r'\b\w+ɓe\b',        # Class 2 (plural person)
        "NC3": r'\b\w+(?:ngel|gel)\b',  # Class 3 (diminutive singular)
        "NC4": r'\b\w+(?:nde|de)\b',    # Class 4 (singular)
        "NC5": r'\b\w+(?:ngu|gu)\b',    # Class 5 (singular)
    }
    
    # Define verb voice suffixes
    voice_suffixes = {
        "ACTIVE": r'\b\w+(?:a|i)\b',     # Active voice
        "MIDDLE": r'\b\w+(?:oo|ike)\b',  # Middle voice
        "PASSIVE": r'\b\w+(?:ee|aama)\b' # Passive voice
    }
    
    # Apply noun class markers
    for nc, pattern in nc_suffixes.items():
        text = re.sub(pattern, f'<{nc}> \g<0>', text)
    
    # Apply verb voice markers
    for voice, pattern in voice_suffixes.items():
        text = re.sub(pattern, f'<{voice}> \g<0>', text)
    
    # Handle consonant mutations
    mutations = {
        'w': 'b', 'b': 'w',  # w/b alternation
        'r': 'd', 'd': 'r',  # r/d alternation
        'y': 'j', 'j': 'y',  # y/j alternation
        'f': 'p', 'p': 'f',  # f/p alternation
        's': 'c', 'c': 's'   # s/c alternation
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word[0].lower() in mutations:
            alt = mutations[word[0].lower()]
            words[i] = f'<ALT_{word[0].upper()}{alt.upper()}> {word}'
    
    return ' '.join(words)

# Test the function
test_text = "Gorko o ƴami nyiiri. Debbo oo looti."
processed_text = apply_grammar_rules(test_text)
print(processed_text)
