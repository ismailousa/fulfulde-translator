import os
from typing import Union, List
from langdetect import detect

# Import the translation function directly from the project's source
try:
    from src.inference.translate import translate_text as base_translate
except ImportError:
    # Fallback implementation if the module cannot be imported
    def base_translate(text, src_lang, tgt_lang, **kwargs):
        return f"[MOCK] Translation from {src_lang} to {tgt_lang}: {text}"

# Import settings
from config.settings import MODEL_PATHS, NLLB_LANG_CODES, M2M100_LANG_CODES

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        # This is a simplified detection - in a real app you might 
        # want to map detected language codes to your supported languages
        detected = detect(text)
        
        # Map detected code to supported languages
        lang_map = {
            "en": "en",
            "fr": "fr",
            "ful": "ff",  # Fulfulde may be detected as 'ful'
        }
        
        return lang_map.get(detected, "en")  # Default to English if not recognized
    except:
        return "en"  # Default to English on error

def translate_text(
    text: Union[str, List[str]], 
    src_lang: str, 
    tgt_lang: str, 
    variant: str = "fine-tuned",
    auto_detect: bool = False
) -> Union[str, List[str]]:
    """
    Translate text using the specified model variant.
    
    Args:
        text: Input text or list of texts to translate
        src_lang: Source language code
        tgt_lang: Target language code
        variant: Model variant ("base" or "fine-tuned")
        auto_detect: Whether to auto-detect the source language
    
    Returns:
        Translated text or list of translations
    """
    if not text:
        return "" if isinstance(text, str) else []
        
    # Auto-detect source language if requested
    if auto_detect and isinstance(text, str):
        src_lang = detect_language(text)
        
    # Get the appropriate model path based on variant
    model_path = MODEL_PATHS.get(variant, MODEL_PATHS["fine-tuned"])
    
    # Determine if using NLLB or M2M100 based on model path
    if "m2m100" in model_path.lower():
        # Use M2M100 language codes
        src_code = M2M100_LANG_CODES.get(src_lang, src_lang)
        tgt_code = M2M100_LANG_CODES.get(tgt_lang, tgt_lang)
    else:
        # Use NLLB language codes (default)
        src_code = NLLB_LANG_CODES.get(src_lang, src_lang)
        tgt_code = NLLB_LANG_CODES.get(tgt_lang, tgt_lang)
        
    # Call the base translation function with appropriate parameters
    try:
        return base_translate(
            text, 
            src_lang=src_code, 
            tgt_lang=tgt_code,
            model_path=model_path
        )
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # Return empty result on error
        return "" if isinstance(text, str) else [""] * len(text)
