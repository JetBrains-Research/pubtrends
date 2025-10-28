# Commands to build container
# docker build -t biolabs/pubtrends-test .
#
# Push to Docker hub
# docker login -u biolabs && docker push biolabs/pubtrends-test

FROM biolabs/pubtrends:latest

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER root

# Install Java 21 and Postgresql 17 & clean apt
RUN echo "deb http://apt.postgresql.org/pub/repos/apt noble-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | \
    sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg \
    && apt-get update --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        openjdk-21-jre-headless \
        postgresql-17 postgresql-client-17 postgresql-contrib-17 \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Configure Postgresql
RUN chmod -R a+w /var/run/postgresql

USER user
# trust authentification method for testing purposes only!
RUN /usr/lib/postgresql/17/bin/initdb -D /home/user/postgres -A trust -U user

## Adjust PostgreSQL configuration so that remote connections to the database are possible.
RUN echo "host all all 0.0.0.0/0 md5" >> /home/user/postgres/pg_hba.conf \
    && echo "listen_addresses='*'" >> /home/user/postgres/postgresql.conf

# Create a PostgreSQL role named `biolabs` with `mysecretpassword` as the password and
# then create a database `pubtrends_test` owned by the `biolabs` role.
RUN /usr/lib/postgresql/17/bin/pg_ctl -D /home/user/postgres start \
    && /usr/lib/postgresql/17/bin/createdb -O user user \
    && psql --command "CREATE ROLE biolabs WITH PASSWORD 'mysecretpassword';" \
    && psql --command "ALTER ROLE biolabs WITH LOGIN;" \
    && psql --command "CREATE DATABASE test_pubtrends OWNER biolabs;" \
    && /usr/lib/postgresql/17/bin/pg_ctl -D /home/user/postgres stop # Stop db

# Expose the PostgreSQL port
EXPOSE 5432

USER user

RUN source activate pubtrends \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install sentence-transformers \
    && pip install faiss-cpu pyarrow fastparquet

# Create nescessary folders
COPY init.sh /home/user/init.sh
RUN bash /home/user/init.sh

# Download NLKT packages
RUN source activate pubtrends \
    && bash /home/user/nlp.sh

# Use `-d` param to launch container as daemon
CMD /usr/lib/postgresql/17/bin/pg_ctl -D /home/user/postgres start && sleep infinity