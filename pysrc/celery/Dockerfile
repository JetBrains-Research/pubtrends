FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

# Configure URLs for between containers communication
ENV CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/0 \
    C_FORCE_ROOT=true \
    EMBEDDINGS_SERVICE_URL=http://embeddings:5001 \
    SEMANTIC_SEARCH_SERVICE_URL=http://semantic:5002 \
    PYTHONPATH=/home/user/pubtrends

# Copy application files
COPY --chown=user:pubtrends pysrc /home/user/pubtrends/pysrc/
COPY --chown=user:pubtrends scripts /home/user/pubtrends/scripts/

WORKDIR /home/user/pubtrends

# Launch celery workers
CMD /bin/bash ~/pubtrends/scripts/nlp.sh \
    && celery --app pysrc.celery.tasks worker -c 2 --loglevel=info --logfile=/logs/celery.log