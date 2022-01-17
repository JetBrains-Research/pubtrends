FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

# Download all the nltk and spacy resources
RUN source activate pubtrends \
    && python -m nltk.downloader averaged_perceptron_tagger punkt stopwords wordnet \
    && python -m spacy download en_core_web_sm

# Configure URLs for between containers communication
ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV FASTTEXT_URL http://fasttext:8081

# Expose the app port, don't use default HTTP 80 or HTTPS 443
EXPOSE 8888

COPY . /home/user/
WORKDIR /home/user/

CMD source activate pubtrends \
    && gunicorn --bind 0.0.0.0:8888 --workers 5 \
        --limit-request-line 0 --timeout=120 \
        --log-level=info --log-file=/logs/web.log \
        --preload "pysrc.app.app:get_app()"
