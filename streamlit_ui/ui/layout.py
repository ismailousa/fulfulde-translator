import streamlit as st

def render_header():
    """Render the app header and description."""
    st.title("🌍 Fulfulde Translator")
    
    # Description and usage tips
    with st.expander("About this translator", expanded=False):
        st.markdown("""
        ## About the Fulfulde Translator
        
        This translator provides bidirectional translation between:
        - Fulfulde (FF) ↔️ English (EN)
        - Fulfulde (FF) ↔️ French (FR)
        
        ### Features
        - Translate text between Fulfulde, English, and French
        - Compare translations from different model variants
        - Submit corrections to help improve the model
        - Express preferences between translation options
        
        ### How to use
        1. Select source and target languages
        2. Enter text to translate
        3. View and compare translations
        4. Provide corrections or select your preferred translation
        
        ### About Fulfulde
        Fulfulde (also known as Fulani) is spoken by 20-25 million people across more than 20 countries in West and Central Africa.
        """)
    
    # Add a separator
    st.markdown("---")
