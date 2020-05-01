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
RUN neo4j-admin set-initial-password password

# Expose Neo4j bolt port
EXPOSE 7687
# Expose Neo4j http interface
EXPOSE 7474

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
    && pip install teamcity-messages pytest-codestyle redis \
    && conda clean -afy \
    && rm /home/user/environment.yml

# Use `-d` param to launch container as daemon
CMD sudo neo4j start && sleep infinity