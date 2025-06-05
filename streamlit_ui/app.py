import streamlit as st
import pandas as pd
from datetime import datetime

# Import UI components
from ui.layout import render_header
from ui.translator_controls import render_translation_controls
from ui.translator_output import render_translation_output

# Import core functionality
from core.translator import translate_text
from core.correction_logger import log_correction
from core.preference_tracker import log_preference

# Import settings
from config.settings import LANGUAGES, DEFAULT_SOURCE, DEFAULT_TARGET, DEFAULT_MODELS

def main():
    # Configure the page
    st.set_page_config(
        page_title="Fulfulde Translator",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Render the header
    render_header()
    
    # Session state initialization
    if "translation" not in st.session_state:
        st.session_state.translation = None
    if "alt_translation" not in st.session_state:
        st.session_state.alt_translation = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "detected_lang" not in st.session_state:
        st.session_state.detected_lang = None
    
    # Create two columns for the translator
    col1, col2 = st.columns([3, 4])

    with col1:
        # Render translation controls (language selectors, input text)
        source_lang, target_lang, input_text, auto_detect, selected_model = render_translation_controls()
    
    # Translate the text when requested
    if input_text and st.session_state.input_text != input_text:
        st.session_state.input_text = input_text
        with st.spinner("Translating..."):
            try:
                # Get translation from primary model
                translation = translate_text(input_text, source_lang, target_lang, variant=selected_model)
                st.session_state.translation = translation
                
                # Get translation from alternative model for comparison
                alt_model = "fine-tuned" if selected_model == "base" else "base"
                alt_translation = translate_text(input_text, source_lang, target_lang, variant=alt_model)
                st.session_state.alt_translation = alt_translation
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
                st.session_state.translation = None
                st.session_state.alt_translation = None
    
    with col2:
        # Render translation output and feedback collection
        if st.session_state.translation:
            # Pass selected model and alternative model names for display
            primary_model = selected_model
            alternative_model = "fine-tuned" if selected_model == "base" else "base"
            
            preferred, correction = render_translation_output(
                input_text, 
                st.session_state.translation, 
                st.session_state.alt_translation, 
                source_lang, 
                target_lang,
                primary_model,
                alternative_model
            )
            
            # Log user preference if they select one of the translation options
            if preferred:
                options = {
                    primary_model: st.session_state.translation,
                    alternative_model: st.session_state.alt_translation
                }
                log_preference(
                    input_text,
                    options,
                    preferred,
                    {
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Log correction if provided
            if correction and correction != st.session_state.translation:
                log_correction(
                    input_text,
                    st.session_state.translation,
                    correction,
                    {
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "model": selected_model,
                        "timestamp": datetime.now().isoformat()
                    }
                )

    # Add a data explorer section in a sidebar
    with st.sidebar:
        st.header("Translation Data")
        
        if st.button("View Collected Corrections"):
            try:
                import pandas as pd
                from utils.jsonl_utils import read_jsonl
                
                corrections = read_jsonl("streamlit_ui/data/corrections.jsonl")
                if corrections:
                    st.dataframe(pd.DataFrame(corrections))
                else:
                    st.info("No corrections collected yet.")
            except Exception as e:
                st.error(f"Error loading corrections: {str(e)}")
        
        if st.button("View Translation Preferences"):
            try:
                import pandas as pd
                from utils.jsonl_utils import read_jsonl
                
                preferences = read_jsonl("streamlit_ui/data/preferences.jsonl")
                if preferences:
                    st.dataframe(pd.DataFrame(preferences))
                else:
                    st.info("No preferences collected yet.")
            except Exception as e:
                st.error(f"Error loading preferences: {str(e)}")

if __name__ == "__main__":
    main()
