# Commands to build container
# docker build -t biolabs/pubtrends-test .
#
# Push to Docker hub
# docker login -u biolabs && docker push biolabs/pubtrends-test

FROM biolabs/pubtrends:latest

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER root

# Install Java 11 and Postgresql 15 & clean apt
ENV TZ Europe/Moscow
RUN echo "deb http://apt.postgresql.org/pub/repos/apt jammy-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add - \
    && apt-get update --fix-missing \
    &&  DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        openjdk-11-jre-headless \
        postgresql-15 postgresql-client-15 postgresql-contrib-15 \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Configure Postgresql
RUN chmod -R a+w /var/run/postgresql

USER user
# trust authentification method for testing purposes only!
RUN /usr/lib/postgresql/15/bin/initdb -D /home/user/postgres -A trust -U user

## Adjust PostgreSQL configuration so that remote connections to the database are possible.
RUN echo "host all all 0.0.0.0/0 md5" >> /home/user/postgres/pg_hba.conf \
    && echo "listen_addresses='*'" >> /home/user/postgres/postgresql.conf

# Create a PostgreSQL role named `biolabs` with `mysecretpassword` as the password and
# then create a database `pubtrends_test` owned by the `biolabs` role.
RUN /usr/lib/postgresql/15/bin/pg_ctl -D /home/user/postgres start \
    && /usr/lib/postgresql/15/bin/createdb -O user user \
    && psql --command "CREATE ROLE biolabs WITH PASSWORD 'mysecretpassword';" \
    && psql --command "ALTER ROLE biolabs WITH LOGIN;" \
    && psql --command "CREATE DATABASE test_pubtrends OWNER biolabs;" \
    # Stop db
    && /usr/lib/postgresql/15/bin/pg_ctl -D /home/user/postgres stop

# Expose the PostgreSQL port
EXPOSE 5432

USER root
RUN mkdir /logs && chmod a+rw /logs \
    && mkdir /database && chmod a+rw /database \
    && mkdir /results && chmod a+rw /results
USER user

# Use `-d` param to launch container as daemon
CMD /usr/lib/postgresql/15/bin/pg_ctl -D /home/user/postgres start && sleep infinity