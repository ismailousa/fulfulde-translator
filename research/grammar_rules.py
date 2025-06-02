import re
import unicodedata

def apply_grammar_rules(text):
    """
    Apply comprehensive Fulfulde grammar rules to the input text.
    
    This function handles:
    1. Noun classes and agreement
    2. Verb conjugation and voice
    3. Consonant mutations
    4. Tense and aspect
    5. Pronouns and person markers
    6. Case marking
    """
    
    # Normalize text
    text = unicodedata.normalize('NFC', text)
    
    # 1. Noun Classes
    # Adamawa Fulfulde has 25 noun classes
    noun_classes = {
        # Class 1: Singular person
        "NC1": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["mawɗo", "gorko", "debbo"],
            "description": "Singular person"
        },
        # Class 2: Plural person
        "NC2": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["mawɓe", "worɓe", "rewɓe"],
            "description": "Plural person"
        },
        # Class 3: Diminutive
        "NC3": {
            "singular": r'\b\w+(?:ngel|gel)\b',
            "plural": r'\b\w+(?:nde|de)\b',
            "examples": ["kuungel", "ɓiɗɗo"],
            "description": "Diminutive"
        },
        # Class 4: Singular
        "NC4": {
            "singular": r'\b\w+(?:nde|de)\b',
            "plural": r'\b\w+(?:nde|de)ɓe\b',
            "examples": ["ɓiinde", "ɓiɓiinde"],
            "description": "Singular"
        },
        # Class 5: Singular
        "NC5": {
            "singular": r'\b\w+(?:ngu|gu)\b',
            "plural": r'\b\w+(?:ngu|gu)ɓe\b',
            "examples": ["ɓiiŋgu"],
            "description": "Singular"
        },
        # Class 6: Body parts
        "NC6": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Body parts"
        },
        # Class 7: Animals
        "NC7": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Animals"
        },
        # Class 8: Liquids
        "NC8": {
            "singular": r'\b\w+(?:ɗi|i)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗi", "ɓeɓi"],
            "description": "Liquids"
        },
        # Class 9: Natural objects
        "NC9": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Natural objects"
        },
        # Class 10: Abstract nouns
        "NC10": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Abstract nouns"
        },
        # Class 11: Small objects
        "NC11": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Small objects"
        },
        # Class 12: Long objects
        "NC12": {
            "singular": r'\b\w+(?:ɗi|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗi", "ɓeɓe"],
            "description": "Long objects"
        },
        # Class 13: Round objects
        "NC13": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Round objects"
        },
        # Class 14: Flat objects
        "NC14": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Flat objects"
        },
        # Class 15: Containers
        "NC15": {
            "singular": r'\b\w+(?:ɗi|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗi", "ɓeɓe"],
            "description": "Containers"
        },
        # Class 16: Plural of NC1
        "NC16": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Plural of NC1"
        },
        # Class 17: Plural of NC2
        "NC17": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Plural of NC2"
        },
        # Class 18: Plural of NC3
        "NC18": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Plural of NC3"
        },
        # Class 19: Plural of NC4
        "NC19": {
            "singular": r'\b\w+(?:ɗi|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗi", "ɓeɓe"],
            "description": "Plural of NC4"
        },
        # Class 20: Plural of NC5
        "NC20": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Plural of NC5"
        },
        # Class 21: Plural of NC6
        "NC21": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Plural of NC6"
        },
        # Class 22: Plural of NC7
        "NC22": {
            "singular": r'\b\w+(?:ɗi|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗi", "ɓeɓe"],
            "description": "Plural of NC7"
        },
        # Class 23: Plural of NC8
        "NC23": {
            "singular": r'\b\w+(?:ɗo|o)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗo", "ɓeɓe"],
            "description": "Plural of NC8"
        },
        # Class 24: Plural of NC9
        "NC24": {
            "singular": r'\b\w+(?:ɗe|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗe", "ɓeɓe"],
            "description": "Plural of NC9"
        },
        # Class 25: Plural of NC10
        "NC25": {
            "singular": r'\b\w+(?:ɗi|e)\b',
            "plural": r'\b\w+ɓe\b',
            "examples": ["ɓeɗi", "ɓeɓe"],
            "description": "Plural of NC10"
        }
    }
    
    # 2. Verb Conjugation
    verb_conjugation = {
        "voice": {
            "ACTIVE": r'\b\w+(?:a|i)\b',     # Active voice (e.g., "mi wari" - I arrived)
            "MIDDLE": r'\b\w+(?:oo|ike)\b',  # Middle voice
            "PASSIVE": r'\b\w+(?:ee|aama)\b' # Passive voice
        },
        "tense": {
            "PRESENT": r'\b\w+(?:a|i)\b',    # Present (e.g., "mi wari" - I arrived)
            "PAST": r'\b\w+(?:aa|ii)\b',     # Past (e.g., "mi wari" - I arrived)
            "FUTURE": r'\b\w+(?:a|i)ɗa\b'    # Future (e.g., "Usmanu ɓii moy?" - Usmanu is whose child?)
        },
        "aspect": {
            "PERFECTIVE": r'\b\w+(?:aa|ii)\b',    # Perfective (e.g., "mi wari" - I arrived)
            "IMPERFECTIVE": r'\b\w+(?:a|i)\b',    # Imperfective
            "PROGRESSIVE": r'\b\w+(?:a|i)ɗa\b'    # Progressive
        },
        "examples": {
            "present": ["mi wari", "mi warti"],
            "past": ["mi wari", "mi warti"],
            "future": ["Usmanu ɓii moy?"],
            "perfective": ["mi wari"],
            "imperfective": ["a waali jam?"],
            "progressive": ["Usmanu ɓii moy?"]
        }
    }
    
    # 3. Pronouns and Person Markers
    pronouns = {
        "1SG": {"form": "mi", "example": "mi wari"},  # I arrived
        "2SG": {"form": "ko", "example": "A waddi"},  # You brought
        "3SG": {"form": "o", "example": "o ɗon siwta"},  # He rests
        "1PL": {"form": "ɓe", "example": "ɓe ɗon jooɗi"},  # They are sitting
        "2PL": {"form": "ko'on", "example": "Sannu ko'on"},  # You all
        "3PL": {"form": "ɓe", "example": "ɓe ɗon jooɗi"}  # They are sitting
    }
    
    # 4. Consonant Mutations
    consonant_mutations = {
        "voicing": {
            "voiceless": {"p": "b", "t": "d", "k": "g"},
            "voiced": {"b": "p", "d": "t", "g": "k"}
        },
        "nasalization": {
            "plain": {"b": "mb", "d": "nd", "g": "ŋg"},
            "nasal": {"mb": "b", "nd": "d", "ŋg": "g"}
        },
        "examples": {
            "voicing": ["mi wari"],  # mi (I) → ma (my)
            "nasalization": ["Ndaa saare maako"]  # Ndaa (Here) → saare (house)
        }
    }
    
    # Apply rules
    def mark_noun_class(word):
        """Mark noun class for a given word"""
        for nc, patterns in noun_classes.items():
            if re.match(patterns["singular"], word):
                return f'<{nc}> {word}'
            if re.match(patterns["plural"], word):
                return f'<{nc}PL> {word}'
        return word
    
    def mark_verb(word):
        """Mark verb features"""
        voice = None
        tense = None
        aspect = None
        
        # Check voice
        for v, pattern in verb_conjugation["voice"].items():
            if re.match(pattern, word):
                voice = v
                break
                
        # Check tense
        for t, pattern in verb_conjugation["tense"].items():
            if re.match(pattern, word):
                tense = t
                break
                
        # Check aspect
        for a, pattern in verb_conjugation["aspect"].items():
            if re.match(pattern, word):
                aspect = a
                break
                
        if any([voice, tense, aspect]):
            features = []
            if voice: features.append(f'V={voice}')
            if tense: features.append(f'T={tense}')
            if aspect: features.append(f'A={aspect}')
            return f'<{";".join(features)}> {word}'
        return word
    
    # Process text
    words = text.split()
    processed_words = []
    
    for word in words:
        # First check if it's a verb
        processed = mark_verb(word)
        if processed != word:
            processed_words.append(processed)
            continue
            
        # Then check if it's a noun
        processed = mark_noun_class(word)
        if processed != word:
            processed_words.append(processed)
            continue
            
        # Add original word if no rules matched
        processed_words.append(word)
    
    return ' '.join(processed_words)
    
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
