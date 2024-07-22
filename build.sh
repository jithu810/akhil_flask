#!/usr/bin/env bash

# Create the directory for NLTK data if it doesn't exist
mkdir -p /opt/render/project/src/nltk_data

# Set the NLTK_DATA environment variable
export NLTK_DATA=/opt/render/project/src/nltk_data

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords', download_dir='/opt/render/project/src/nltk_data')"
