# Commands to build container
# docker build -t biolabs/pubtrends .
#
# Push to Docker hub
# docker login -u biolabs && docker push biolabs/pubtrends

FROM python:3.10-slim

LABEL author="Oleg Shpynov"
LABEL email="os@jetbrains.com"

# Install only essential packages and cleanup in one layer
RUN DEBIAN_FRONTEND="noninteractive" apt-get update --fix-missing \
    && apt-get install --no-install-recommends -y \
        curl ca-certificates gcc g++ libpq-dev \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make new user
RUN groupadd -r pubtrends && useradd -ms /bin/bash -g pubtrends user

# Install uv and Python dependencies
USER user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:$PATH"

# Install required dependencies from pyproject.toml
COPY --chown=user:pubtrends pyproject.toml /home/user/pyproject.toml
WORKDIR /home/user
RUN uv venv /home/user/.venv \
    && uv pip install --no-cache pip tomli \
    && /home/user/.venv/bin/python -c "import tomli; print('\n'.join(tomli.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/deps.txt \
    && cat /tmp/deps.txt \
    && uv pip install --no-cache -r /tmp/deps.txt \
    && rm -rf /home/user/.cache /tmp/deps.txt

ENV PATH="/home/user/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/user/.venv"
ENV PYTHONUNBUFFERED=1
