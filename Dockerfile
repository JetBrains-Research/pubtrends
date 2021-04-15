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
    && apt-get install -y curl bzip2 gnupg2 wget ca-certificates sudo default-jre

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
    && pip install teamcity-messages pytest-flake8 \
    && conda clean -afy \
    && rm /home/user/environment.yml

# Download nltk & spacy resources
RUN source activate pubtrends \
    && python -m nltk.downloader averaged_perceptron_tagger punkt stopwords wordnet \
    && python -m spacy download en_core_web_sm

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

USER root
RUN mkdir /logs && chmod a+rw /logs \
    && mkdir /database && chmod a+rw /database \
    && mkdir /predefined && chmod a+rw /predefined
USER user

# Use `-d` param to launch container as daemon
CMD /usr/lib/postgresql/12/bin/pg_ctl -D /home/user/postgres start && sleep infinity