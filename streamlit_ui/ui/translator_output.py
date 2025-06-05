import streamlit as st

def render_translation_output(
    input_text, 
    translation, 
    alt_translation, 
    source_lang, 
    target_lang,
    primary_model,
    alternative_model
):
    """
    Render the translation output and feedback collection.
    
    Args:
        input_text: The original input text
        translation: The primary translation
        alt_translation: The alternative translation
        source_lang: Source language code
        target_lang: Target language code
        primary_model: Name of the primary model
        alternative_model: Name of the alternative model
        
    Returns:
        tuple: (preferred_model, user_correction)
    """
    st.subheader("Translation Results")
    
    # Initialize return values
    preferred_model = None
    user_correction = None
    
    # Display the translation with a copy button
    st.markdown("### Primary Translation")
    st.markdown(f"*Using {primary_model} model:*")
    
    st.code(translation, language=None)
    
    # Add button to copy to clipboard
    if st.button("📋 Copy to clipboard", key="copy_main"):
        st.write("Copied to clipboard! (frontend simulation)")
        
    # Display the alternative translation if available
    if alt_translation and alt_translation != translation:
        st.markdown("### Alternative Translation")
        st.markdown(f"*Using {alternative_model} model:*")
        
        st.code(alt_translation, language=None)
        
        # Add button to copy alternative to clipboard
        if st.button("📋 Copy to clipboard", key="copy_alt"):
            st.write("Copied to clipboard! (frontend simulation)")
        
        # Preference selection
        st.markdown("### Which translation do you prefer?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Prefer {primary_model}"):
                st.success(f"You preferred the {primary_model} translation")
                preferred_model = primary_model
        
        with col2:
            if st.button(f"Prefer {alternative_model}"):
                st.success(f"You preferred the {alternative_model} translation")
                preferred_model = alternative_model
    
    # Correction input
    st.markdown("### Suggest a correction")
    st.markdown(
        "If the translation isn't accurate, please provide a corrected version to help improve the model:"
    )
    
    correction = st.text_area(
        "Corrected translation",
        value=translation,
        height=100
    )
    
    # Submit correction button
    if st.button("Submit Correction"):
        if correction != translation:
            st.success("Thank you for your correction! It will help improve future translations.")
            user_correction = correction
        else:
            st.info("No changes were made to the translation.")
    
    return preferred_model, user_correction
