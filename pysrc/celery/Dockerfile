FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

# Update pubtrends
RUN source activate pubtrends && pip install redis

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

COPY . /home/user/
WORKDIR /home/user/

# Download all the nltk resources
RUN source activate pubtrends && python pysrc/papers/utils.py

# Launch celery workers
ENTRYPOINT source activate pubtrends \
    && celery -A pysrc.celery.tasks worker -c 5 --loglevel=info --logfile=/logs/celery.log