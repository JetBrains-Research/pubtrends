FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

# Download all the nltk and spacy resources
COPY nlp.sh /home/user/nlp.sh
RUN source activate pubtrends \
    && python -m spacy download en_core_web_sm

# Configure URLs for between containers communication
ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV EMBEDDINGS_SERVICE_URL=http://embeddings:5001
ENV SEMANTIC_SEARCH_SERVICE_URL=http://semantic:5002

# Expose the app port, don't use default HTTP 80 or HTTPS 443
EXPOSE 5000

RUN mkdir -p /home/user/pubtrends/pysrc
COPY pysrc /home/user/pubtrends/pysrc/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc

USER user
WORKDIR /home/user/pubtrends

CMD source activate pubtrends \
    && export PYTHONPATH=/home/user/pubtrends \
    && gunicorn --bind 0.0.0.0:5000 --workers 5 \
        --limit-request-line 0 --timeout=120 \
        --log-level=info --log-file=/logs/gunicorn.log \
        --preload "pysrc.app.pubtrends_app:get_app()"
