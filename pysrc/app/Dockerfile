FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER user

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

ENV HOST 0.0.0.0
ENV PORT 8000
ENV DEBUG true

# expose the app port
EXPOSE 8000

COPY . /home/user/
WORKDIR /home/user/

# Download spacy corpus
RUN source activate pubtrends && python -m spacy download en_core_web_sm

# run the app server
CMD echo $PUBTRENDS_VERSION \
    && source activate pubtrends \
    && gunicorn --bind 0.0.0.0:8000 --workers 5 \
        --log-level=info --log-file=/logs/gunicorn.log "pysrc.app.app:get_app()"
