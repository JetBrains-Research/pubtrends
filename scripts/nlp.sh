#!/usr/bin/env bash
# Download all the nltk resources
python -c "import nltk; [nltk.download(p, raise_on_error=True) for p in ['punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt', 'stopwords', 'wordnet', 'omw-1.4']]"
# Spacy model is now installed via pyproject.toml