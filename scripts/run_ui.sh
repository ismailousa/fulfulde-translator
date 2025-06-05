#!/bin/bash
# Script to run the Streamlit UI using Poetry

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Set environment variables if needed
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the Streamlit app with Poetry
poetry run streamlit run streamlit_ui/app.py "$@"
