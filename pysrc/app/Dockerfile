FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

# Configure URLs for between containers communication
ENV CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/0 \
    EMBEDDINGS_SERVICE_URL=http://embeddings:5001 \
    SEMANTIC_SEARCH_SERVICE_URL=http://semantic:5002 \
    PYTHONPATH=/home/user/pubtrends

EXPOSE 5000

# Copy application files
COPY --chown=user:pubtrends pysrc /home/user/pubtrends/pysrc/
COPY --chown=user:pubtrends scripts /home/user/pubtrends/scripts/

WORKDIR /home/user/pubtrends

CMD /bin/bash ~/pubtrends/scripts/nlp.sh \
    && gunicorn --bind 0.0.0.0:5000 --workers 5 \
        --limit-request-line 0 --timeout=300 \
        --log-level=info --log-file=/logs/gunicorn.log \
        --preload "pysrc.app.pubtrends_app:get_app()"
