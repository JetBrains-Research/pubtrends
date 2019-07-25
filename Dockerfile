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

# On Dockerhub, the new Ubuntu 18.04 LTS image is now the new Minimal Ubuntu 18.04 image.
FROM ubuntu:18.04

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

USER root

# Update all the packages
RUN apt-get update --fix-missing

# Install conda, curl should install certificates, so no --no-install-recommends
RUN apt-get install -y curl bzip2
RUN curl --location https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --output ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN ln -snf /bin/bash /bin/sh

# Install Postgresql, Redis and cleanup
RUN apt-get install --no-install-recommends -y postgresql postgresql-contrib \
    && apt-get install --no-install-recommends -y redis-server \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Make new user
RUN groupadd -r pubtrends && useradd -ms /bin/bash -g pubtrends user

# Create pubtrends conda env
USER user
COPY environment.yml /home/user/environment.yml
RUN conda init bash \
    && conda env create -f /home/user/environment.yml \
    && source activate pubtrends \
    && pip install teamcity-messages pytest-codestyle \
    && conda clean -afy \
    && source deactivate \
    && rm /home/user/environment.yml

# Create Postgresql cluster
USER root
RUN chmod -R a+w /var/run/postgresql

USER user
# trust authentification method for testing purposes only!
RUN /usr/lib/postgresql/10/bin/initdb -D /home/user/postgres -A trust -U user

## Adjust PostgreSQL configuration so that remote connections to the database are possible.
RUN echo "host all all 0.0.0.0/0 md5" >> /home/user/postgres/pg_hba.conf \
    && echo "listen_addresses='*'" >> /home/user/postgres/postgresql.conf

# Expose the PostgreSQL port
EXPOSE 5432

# Create a PostgreSQL role named `biolabs` with `password` as the password and
# then create a database `pubtrends_test` owned by the `biolabs` role.
RUN /usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres start \
    && /usr/lib/postgresql/10/bin/createdb -O user user \
    && psql --command "CREATE ROLE biolabs WITH PASSWORD 'password';" \
    && psql --command "ALTER ROLE biolabs WITH LOGIN;" \
    && psql --command "CREATE DATABASE pubtrends_test OWNER biolabs;" \
    # Stop db
    && /usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres stop

# Tests project configuration
COPY config.properties /home/user/config.properties
RUN mkdir -p ~/.pubtrends \
    && cat ~/config.properties | sed 's/5433/5432/g' > ~/.pubtrends/config.properties \
    && rm /home/user/config.properties

# Use `-d` param to launch container as daemon with Postgresql running
CMD /usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres start \
    && sleep infinity