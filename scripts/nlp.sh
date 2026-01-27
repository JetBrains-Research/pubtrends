# Download all the nltk resources
python -m nltk.downloader -d ~/nltk_data \
      punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng punkt stopwords wordnet omw-1.4
# Spacy model is now installed via pyproject.toml