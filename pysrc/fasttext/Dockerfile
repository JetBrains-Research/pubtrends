FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

# Download all the nltk and spacy resources
RUN source activate pubtrends \
    && python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng \
           punkt punkt_tab stopwords wordnet omw-1.4 \
           && python -m spacy download en_core_web_sm

COPY . /home/user/
WORKDIR /home/user/

EXPOSE 5001

CMD source activate pubtrends \
   && gunicorn --bind 0.0.0.0:5001 --workers 1 --threads 4 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/fasttext_gunicorn.log \
       --preload "pysrc.fasttext.fasttext_app:get_app()"
