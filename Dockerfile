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

# Update all the packages, install Redis and cleanup
RUN apt-get update --fix-missing \
    && apt-get install --no-install-recommends -y curl ca-certificates redis-server \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Make new user
RUN groupadd -r pubtrends && useradd -ms /bin/bash -g pubtrends user

# Install Conda locally and create pubtrends conda env
USER user
RUN curl --location https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --output ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b \
    && rm ~/miniconda.sh
ENV PATH /home/user/miniconda3/bin:$PATH
# Fix shell for conda
USER root
RUN ln -snf /bin/bash /bin/sh
USER user

COPY environment.yml /home/user/environment.yml
RUN conda init bash \
    && conda env create -f /home/user/environment.yml \
    && source activate pubtrends \
    && pip install teamcity-messages pytest-codestyle redis \
    && conda clean -afy \
    && rm /home/user/environment.yml

# Tests project configuration
COPY config.properties /home/user/config.properties
RUN mkdir -p ~/.pubtrends \
    && cat ~/config.properties > ~/.pubtrends/config.properties \
    && rm /home/user/config.properties
