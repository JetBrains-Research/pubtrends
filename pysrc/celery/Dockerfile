FROM biolabs/pubtrends

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER user

# Download all the nltk and spacy resources
COPY nlp.sh /home/user/nlp.sh
RUN /bin/bash ~/nlp.sh

# Configure URLs for between containers communication
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0
ENV C_FORCE_ROOT=true
ENV EMBEDDINGS_SERVICE_URL=http://embeddings:5001

RUN mkdir -p /home/user/pubtrends/pysrc
COPY pysrc /home/user/pubtrends/pysrc/

USER root
RUN chown -R user:pubtrends /home/user/pubtrends/pysrc

USER user
WORKDIR /home/user/pubtrends


# Launch celery workers
CMD source activate pubtrends \
    && export PYTHONPATH=/home/user/pubtrends \
    && celery --app pysrc.celery.tasks worker -c 4 --loglevel=info --logfile=/logs/celery.log