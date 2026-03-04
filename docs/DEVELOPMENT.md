# Development Guide

## Prerequisites

Please ensure that you have a database configured, up and running. See [DATABASE.md](DATABASE.md) for database setup instructions.

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`.
   Ensure that file contains correct information about the database(s) (url, port, DB name, username and password).

2. Python environment `pubtrends` can be easily created using uv for launching Jupyter Notebook and Web Service:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r pyproject.toml
```

3. Build base Docker image `biolabs/pubtrends` and nested image `biolabs/pubtrends-test` for testing:

```bash
docker build -f resources/docker/main/Dockerfile -t biolabs/pubtrends --platform linux/amd64  .
docker build  -f resources/docker/test/Dockerfile -t biolabs/pubtrends-test --platform linux/amd64 .
```

## Docker Images

Two Docker images are used for testing, development and deployment:
* [biolabs/pubtrends](resources/docker/main/Dockerfile) - production
* [biolabs/pubtrends-test](resources/docker/test/Dockerfile) - testing

We use [Docker Hub](https://hub.docker.com/) to store built images.

## Kotlin/Java Build

Use the following command to test and build the JAR package:

```bash
./gradlew clean test shadowJar
```

## Web Service

1. Create the necessary folders with script `scripts/init.sh` and download prerequisites:

```bash
bash scripts/init.sh
bash scripts/nlp.sh
```

2. Start Redis:

```bash
docker run -p 6379:6379 redis:7.4.2
```

3. Configure Python environment with uv:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r pyproject.toml
pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache sentence-transformers faiss-cpu jupyter notebook
```

4. Start Celery worker queue:

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
celery -A pysrc.celery.tasks worker -c 1 --loglevel=debug
```

5. Start flask server at http://localhost:5000/:

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pysrc.app.pubtrends_app
```

6. Start service for text embeddings based on either pretrained fasttext model or sentence-transformer at http://localhost:5001/:

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pysrc.endpoints.embeddings.fasttext.fasttext_app
```

or

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer_app
```

7. Optionally, start a semantic search service http://localhost:5002/:

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pysrc.endpoints.semantic_search.semantic_search_app
```

## API Documentation

PubTrends provides interactive API documentation using Swagger UI. Once the Flask server is running, you can access the API documentation at:

* **Swagger UI**: http://localhost:5000/swagger

The Swagger interface provides:
* Interactive API endpoint exploration
* Request/response schema documentation
* Ability to test API endpoints directly from the browser
* Detailed parameter descriptions and examples

## Jupyter Notebook

Notebooks are located under the `/notebooks` folder. Please configure `PYTHONPATH` before using jupyter.

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
jupyter notebook
```

## Testing

### 1. Start Test Database

Start a Docker image with a Postgres environment for tests (Kotlin and Python development):

```bash
docker run --rm --platform linux/amd64 --name pubtrends-test \
--publish=5433:5432 --volume=$(pwd):/pubtrends -i -t biolabs/pubtrends-test
```

**NOTE**: remember to stop the container afterward.

### 2. Kotlin Tests

```bash
./gradlew clean test
```

### 3. Python Tests

Python tests with code style check for development (including integration with Kotlin DB writers):

```bash
source .venv/bin/activate; pytest pysrc
```

### 4. Python Tests within Docker

Ensure that `./build/libs/pubtrends-dev.jar` file is present:

```bash
docker run --rm --platform linux/amd64 --volume=$(pwd):/pubtrends -t biolabs/pubtrends-test /bin/bash -c \
"/usr/lib/postgresql/17/bin/pg_ctl -D /home/user/postgres start; \
cd /pubtrends; cp config.properties /home/user/.pubtrends/; \
pytest pysrc"
```

## Deployment

Deployment is done with docker-compose:

* Gunicorn serving main pubtrends Flask app
* Redis as a message proxy
* Celery workers queue

Please ensure that you have configured and prepared the database(s). See [DATABASE.md](DATABASE.md) for details.

### Deployment Steps

1. Modify file `config.properties` with information about the database(s). File from the project folder is used in this case.

2. Build ready for deployment package with script `scripts/dist.sh`:

```bash
scripts/dist.sh build=build-number ga=google-analytics-id
```

3. Launch pubtrends with docker-compose (one of the options):

```bash
# start with local word2vec tf-idf tokens embeddings
docker-compose -f docker-compose/word2vec.yml up --build

# start with BioWord2Vec tokens embeddings
docker-compose -f docker-compose/fasttext.yml up --build

# start with Sentence Transformer for text embeddings
docker-compose -f docker-compose/sentence-transformer.yml up --build

# Start with Semantic Search based on Sentence Transformer
docker-compose -f docker-compose/semantic-search.yml up --build
```

Use these commands to stop compose build and check logs:

```bash
# stop
docker-compose -f docker-compose/semantic-search.yml down --remove-orphans
# inspect logs
docker-compose -f docker-compose/semantic-search.yml logs
```

Pubtrends will be serving on port 5000.

4. Update nginx timeouts:

```nginx
# increase timeouts
proxy_connect_timeout 60s;
proxy_send_timeout    600s;
proxy_read_timeout    600s;
send_timeout          600s;
```

## Maintenance

Use a simple placeholder during maintenance:

```bash
cd pysrc/app; python -m http.server 5000
```

## Release

* Update `docs/CHANGES.md`
* Update version in `scripts/dist.sh`
* Launch `scripts/dist.sh`, `pubtrends-XXX.tar.gz` will be created in the `dist` directory.
