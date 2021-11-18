# Commands to build container
# docker build -t biolabs/pubtrends .
#
# Push to Docker hub
# docker login -u biolabs && docker push biolabs/pubtrends

# On Dockerhub, Ubuntu 20.04 LTS image is now the new Minimal Ubuntu image.
FROM ubuntu:20.04

LABEL author = "Oleg Shpynov"
LABEL email = "os@jetbrains.com"

# Install required packages and cleanup
RUN DEBIAN_FRONTEND="noninteractive" apt-get update --fix-missing \
    && apt-get install --no-install-recommends -y \
        build-essential curl bzip2 gnupg2 wget ca-certificates sudo \
    && apt-get clean \
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
RUN conda install -c conda-forge mamba

# Install required dependencies
COPY environment.yml /home/user/environment.yml
RUN mamba env update -f /home/user/environment.yml \
    && mamba clean -afy