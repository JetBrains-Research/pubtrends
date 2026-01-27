# Commands to build container
# docker build -t biolabs/pubtrends-test .
#
# Push to Docker hub
# docker login -u biolabs && docker push biolabs/pubtrends-test

FROM biolabs/pubtrends:latest

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

USER root

# Install Java 21 and Postgresql 17 & clean apt in one layer
RUN apt-get update --fix-missing \
    && DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        gnupg2 lsb-release wget \
    && echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg \
    && mkdir -p /etc/apt/keyrings \
    && wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/adoptium.list \
    && apt-get update --fix-missing \
    && DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        temurin-21-jre \
        postgresql-17 postgresql-client-17 postgresql-contrib-17 postgresql-17-pgvector \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

# Install additional dependencies (use absolute path since ENV PATH may not expand correctly)
RUN /home/user/.local/bin/uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu \
   && /home/user/.local/bin/uv pip install --no-cache sentence-transformers faiss-cpu \
   && rm -rf /home/user/.cache

# Create nescessary folders and download NLTK packages
COPY --chown=user:pubtrends scripts /home/user/pubtrends/scripts/
RUN bash /home/user/pubtrends/scripts/init.sh \
    && bash /home/user/pubtrends/scripts/nlp.sh \
    && rm -rf /home/user/.cache

# Use `-d` param to launch container as daemon
CMD /usr/lib/postgresql/17/bin/pg_ctl -D /home/user/postgres start && sleep infinity