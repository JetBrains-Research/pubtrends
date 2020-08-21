# Commands to build container
#
# docker build -t biolabs/pubtrends .
#
# Before push you have to login to docker hub first.
#
# docker login -u biolabs
#
# Then you just push current image
#
# docker push biolabs/pubtrends

# On Dockerhub, Ubuntu 20.04 LTS image is now the new Minimal Ubuntu image.
FROM ubuntu:20.04

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER root

# Update all the packages
RUN apt-get update --fix-missing \
    && apt-get install -y curl bzip2 gnupg2 wget ca-certificates sudo

# Install neo4j
RUN wget --quiet --no-check-certificate -O - https://debian.neo4j.org/neotechnology.gpg.key | apt-key add - \
    && echo 'deb http://debian.neo4j.org/repo stable/' > /etc/apt/sources.list.d/neo4j.list \
    && apt-get update && apt-get install --no-install-recommends -y neo4j

# Install neo4j APOC library
RUN wget --quiet --no-check-certificate -P /var/lib/neo4j/plugins \
    https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/3.5.0.4/apoc-3.5.0.4-all.jar

# Configure neo4j
RUN sed -i "s/#dbms.connectors.default_listen_address=0.0.0.0/dbms.connectors.default_listen_address=0.0.0.0/g" /etc/neo4j/neo4j.conf \
    && sed -i "s/#dbms.security.allow_csv_import_from_file_urls=true/dbms.security.allow_csv_import_from_file_urls=true/g" /etc/neo4j/neo4j.conf \
    && sed -i "s#dbms.directories.import=/var/lib/neo4j/import#dbms.directories.import=/pubtrends#g" /etc/neo4j/neo4j.conf \
    && sed -i "s/#dbms.security.procedures.unrestricted=my.extensions.example,my.procedures.*/dbms.security.procedures.unrestricted=apoc.*/g" /etc/neo4j/neo4j.conf

# Initial password for neo4j user
RUN neo4j-admin set-initial-password mysecretpassword

# Expose Neo4j bolt port
EXPOSE 7687
# Expose Neo4j http interface
EXPOSE 7474

# Install Postgresql 12
ENV TZ Europe/Moscow
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y postgresql-12 postgresql-client-12 postgresql-contrib-12

# Clean apt
RUN apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Make new user
RUN groupadd -r pubtrends && useradd -ms /bin/bash -g pubtrends user && usermod -aG sudo user
# Sudo without password
RUN echo "user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install Conda and create pubtrends conda env
USER user
RUN curl --location https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --output ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b \
    && rm ~/miniconda.sh
ENV PATH /home/user/miniconda3/bin:$PATH
# Fix shell for conda
USER root
RUN ln -snf /bin/bash /bin/sh

USER user
# Create pubtrends conda env
COPY environment.yml /home/user/environment.yml
RUN conda init bash \
    && conda env create -f /home/user/environment.yml \
    && source activate pubtrends \
    && pip install teamcity-messages pytest-pycodestyle \
    && conda clean -afy \
    && rm /home/user/environment.yml

# Configure Postgresql configuration
USER root
RUN chmod -R a+w /var/run/postgresql

USER user
# trust authentification method for testing purposes only!
RUN /usr/lib/postgresql/12/bin/initdb -D /home/user/postgres -A trust -U user

## Adjust PostgreSQL configuration so that remote connections to the database are possible.
RUN echo "host all all 0.0.0.0/0 md5" >> /home/user/postgres/pg_hba.conf \
    && echo "listen_addresses='*'" >> /home/user/postgres/postgresql.conf

# Create a PostgreSQL role named `biolabs` with `mysecretpassword` as the password and
# then create a database `pubtrends_test` owned by the `biolabs` role.
RUN /usr/lib/postgresql/12/bin/pg_ctl -D /home/user/postgres start \
    && /usr/lib/postgresql/12/bin/createdb -O user user \
    && psql --command "CREATE ROLE biolabs WITH PASSWORD 'mysecretpassword';" \
    && psql --command "ALTER ROLE biolabs WITH LOGIN;" \
    && psql --command "CREATE DATABASE test_pubtrends OWNER biolabs;" \
    # Stop db
    && /usr/lib/postgresql/12/bin/pg_ctl -D /home/user/postgres stop

# Expose the PostgreSQL port
EXPOSE 5432

# Use `-d` param to launch container as daemon
CMD /usr/lib/postgresql/12/bin/pg_ctl -D /home/user/postgres start && sudo neo4j start && sleep infinity