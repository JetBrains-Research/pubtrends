FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

# Download all the nltk and spacy resources
RUN source activate pubtrends \
    && python -m nltk.downloader averaged_perceptron_tagger punkt stopwords wordnet \
    && python -m spacy download en_core_web_sm

COPY . /home/user/
WORKDIR /home/user/


CMD source activate pubtrends \
   && gunicorn --bind 0.0.0.0:8081 --workers 1 --threads 4 --worker-class=gthread \
       --limit-request-line 0 --timeout=120 \
       --log-level=info --log-file=/logs/fasttext_web.log \
       --preload "pysrc.fasttext.fasttext_app:get_app()"
