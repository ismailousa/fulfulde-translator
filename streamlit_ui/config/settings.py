# Available languages for translation
LANGUAGES = {
    "ff": "Fulfulde",
    "en": "English",
    "fr": "French"
}

# Default language settings
DEFAULT_SOURCE = "ff"
DEFAULT_TARGET = "en"

# Available model variants
DEFAULT_MODELS = ["base", "fine-tuned"]
DEFAULT_MODEL = "fine-tuned"

# Language codes mapping (for different model types)
NLLB_LANG_CODES = {
    "ff": "ff",
    "en": "en",
    "fr": "fr"
}

M2M100_LANG_CODES = {
    "ff": "ff_Latn",
    "en": "en_Latn",
    "fr": "fr_Latn"
}

# Model paths
MODEL_PATHS = {
    "base": "models/nllb_base",
    "fine-tuned": "models/nllb_ff_fr"
}
