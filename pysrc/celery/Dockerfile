FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

COPY . /home/user/
WORKDIR /home/user/

# Download spacy corpus
RUN source activate pubtrends && python -m spacy download en_core_web_sm

# Launch celery workers
ENTRYPOINT source activate pubtrends \
    && celery -A pysrc.celery.tasks worker -c 1 --loglevel=debug --logfile=/logs/celery.log