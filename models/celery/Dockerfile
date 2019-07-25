FROM biolabs/pubtrends

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

ARG POSTGRESQL_URL
ARG POSTGRESQL_PORT
ARG POSTGRESQL_USERNAME
ARG POSTGRESQL_PASSWORD
ARG POSTGRESQL_DATABASE

# Update pubtrends
RUN source activate pubtrends && pip install redis && source deactivate


ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true


COPY . /home/user/
WORKDIR /home/user/

# Config file
RUN cat config.properties_example \
    | sed "s/url = localhost/url = $POSTGRESQL_URL/g" \
    | sed "s/port = 5432/port = $POSTGRESQL_PORT/g" \
    | sed "s/username = biolabs/username = $POSTGRESQL_USERNAME/g" \
    | sed "s/password = password/password = $POSTGRESQL_PASSWORD/g" \
    | sed "s/database = pubtrends/database = $POSTGRESQL_DATABASE/g" \
    > ~/.pubtrends/config.properties

# Launch celery workers
ENTRYPOINT source activate pubtrends && celery -A models.celery.tasks worker --loglevel=info