# Download all the nltk and spacy resources
source activate pubtrends \
    && python -m nltk.downloader punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng \
           punkt stopwords wordnet omw-1.4 \
           && python -m spacy download en_core_web_sm
