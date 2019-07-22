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

USER root

# Update all the packages
RUN apt-get update --fix-missing

# Install conda
RUN apt-get install -y curl bzip2
RUN curl --location https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --output ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN ln -snf /bin/bash /bin/sh

# Install Postgresql and start
RUN apt-get install -y postgresql postgresql-contrib

# Install Redis
RUN apt-get install -y redis-server

# Make new user
RUN groupadd -r pubtrends && useradd -ms /bin/bash -g pubtrends user

USER user
# Init conda
RUN conda init bash

# Conda envs
COPY environment.yml /home/postgres/environment.yml
RUN conda env create -f /home/postgres/environment.yml
RUN source activate pubtrends && pip install teamcity-messages pytest-codestyle && source deactivate

# Create Postgresql cluster
USER root
RUN chmod -R a+w /var/run/postgresql
USER user
# trust authentification method for testing purposes only!
RUN /usr/lib/postgresql/10/bin/initdb -D /home/user/postgres -A trust -U user

## Adjust PostgreSQL configuration so that remote connections to the database are possible.
RUN echo "host all  all    0.0.0.0/0  md5" >> /home/user/postgres/pg_hba.conf
RUN echo "listen_addresses='*'" >> /home/user/postgres/postgresql.conf

# Expose the PostgreSQL port
EXPOSE 5432

# Create a PostgreSQL role named `biolabs` with `password` as the password and
# then create a database `pubtrends_test` owned by the `biolabs` role.
RUN /usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres start &&\
    /usr/lib/postgresql/10/bin/createdb -O user user &&\
    psql --command "CREATE ROLE biolabs WITH PASSWORD 'password';" &&\
    psql --command "ALTER ROLE biolabs WITH LOGIN;" &&\
    psql --command "CREATE DATABASE pubtrends_test OWNER biolabs;"

# Stop not required
#RUN /usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres stop

RUN mkdir -p ~/.pubtrends

# Test database configuration
COPY config.properties_example /home/user/config.properties_example
RUN cat ~/config.properties_example | sed 's/5433/5432/g' > ~/.pubtrends/config.properties

# Use `-d` param to launch container as daemon with Postgresql running
CMD /usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres start; sleep infinity