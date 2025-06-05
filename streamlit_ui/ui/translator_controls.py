import streamlit as st
from config.settings import LANGUAGES, DEFAULT_SOURCE, DEFAULT_TARGET, DEFAULT_MODELS, DEFAULT_MODEL

def render_translation_controls():
    """
    Render the translation controls (language selectors, input text area).
    
    Returns:
        tuple: (source_lang, target_lang, input_text, auto_detect, selected_model)
    """
    # Language selection
    st.subheader("Translation Settings")
    
    col1, col2, col3 = st.columns([1, 0.2, 1])
    
    with col1:
        source_lang = st.selectbox(
            "Source Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            index=list(LANGUAGES.keys()).index(DEFAULT_SOURCE) if DEFAULT_SOURCE in LANGUAGES else 0
        )
    
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 32px;'>↔️</div>", unsafe_allow_html=True)
        if st.button("🔄", help="Swap languages"):
            # Get current values
            current_source = source_lang
            current_target = st.session_state.get("target_lang", DEFAULT_TARGET)
            
            # Store swapped values in session state
            st.session_state["source_lang"] = current_target
            st.session_state["target_lang"] = current_source
            
            # Force a rerun to update the UI
            st.experimental_rerun()
    
    with col3:
        target_lang = st.selectbox(
            "Target Language",
            options=[lang for lang in LANGUAGES.keys() if lang != source_lang],
            format_func=lambda x: LANGUAGES[x],
            index=0,
            key="target_lang"
        )
    
    # Auto-detect option
    auto_detect = st.checkbox("Auto-detect source language", value=False)
    
    # Model selection
    selected_model = st.selectbox(
        "Select model variant",
        options=DEFAULT_MODELS,
        index=DEFAULT_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in DEFAULT_MODELS else 0
    )
    
    # Input text area
    st.subheader("Enter Text to Translate")
    input_text = st.text_area(
        "Input Text",
        height=150,
        key="input_text_area",
        value=st.session_state.get("input_text", "")
    )
    
    # Clear button
    if st.button("Clear"):
        st.session_state.input_text = ""
        st.session_state.translation = None
        st.session_state.alt_translation = None
        # Force rerun to clear the UI
        st.experimental_rerun()
    
    # Return values
    return source_lang, target_lang, input_text, auto_detect, selected_model
