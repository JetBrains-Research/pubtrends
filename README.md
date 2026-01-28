[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Build Status](http://teamcity.jetbrains.com/app/rest/builds/buildType:(id:BioLabs_Pubtrends_Deployment)/statusIcon.svg)](http://teamcity.jetbrains.com/viewType.html?buildTypeId=BioLabs_Pubtrends_Deployment&guest=1)
[![DOI](https://zenodo.org/badge/151591143.svg)](https://doi.org/10.5281/zenodo.15131474)

PubTrends
=========

PubTrends is an interactive scientific literature exploration tool that helps researchers analyze topics, visualize
research trends, and discover related works.

Available online at: https://pubtrends.info/

## Overview
With PubTrends, you can:

* Gain a concise overview of your research area.
* Explore popular trends and impactful publications.
* Discover new and promising research directions.

See example of analysis at: https://pubtrends.info/about.html

## Datasets:

* [Pubmed](https://pubmed.ncbi.nlm.nih.gov) ~40 mln papers and 450 mln citations
* [Semantic Scholar](https://www.semanticscholar.org) 170 mln papers and 600 mln citations

![Scheme](pysrc/app/static/about/about_pubtrends_scheme.png?raw=true "Title")

## Technical Architecture

PubTrends is a Python / Kotlin + JavaScript web service with a PostgreSQL backend.
It uses:
* Languages: [Python](https://www.python.org/) + [Kotlin](https://kotlinlang.org/) + [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
* Backend: [Nginx](https://nginx.org/en/index.html) + [Flask](https://flask.palletsprojects.com/en/stable/) + [Gunicorn](https://gunicorn.org/)
* Task Queue: [Celery](https://docs.celeryq.dev/en/stable/index.html) + [Redis](https://redis.io/)
* DataBase: [Postgres](https://www.postgresql.org/) + [Postgres pgvector](https://github.com/pgvector/pgvector) + [Psycopg2](https://www.psycopg.org/docs/) + [Kotlin ORM](https://www.jetbrains.com/exposed/)
* Data Analysis: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/stable/index.html#)
* Semantic Search: [Sentence-Tranformers](https://www.sbert.net/) + [Faiss](https://github.com/facebookresearch/faiss)
* NLP: [NLTK](https://www.nltk.org/), [SpaCy](https://spacy.io/), [GenSim](https://radimrehurek.com/gensim/models/word2vec.html), [Fasttext](https://fasttext.cc/)
* Visualization: [Bokeh](https://bokeh.org/), [Holoviews](https://holoviews.org/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)
* Frontend: [Bootstrap](https://getbootstrap.com/), [jQuery](https://jquery.com/), [Cytoscape.js](https://js.cytoscape.org/)
* Deployment: [Docker Compose](https://docs.docker.com/compose/)
* Testing: [PyTest](https://docs.pytest.org/en/stable/) + [Flake8](https://flake8.pycqa.org/en/latest/) + [JUnit](https://junit.org/) + [TeamCity](https://www.jetbrains.com/teamcity/)

See [pyproject.toml](pyproject.toml) for the full list of libraries used in the project.

## Docker

Two Docker images are used for testing and deployment: 
* [biolabs/pubtrends](resources/docker/main/Dockerfile) - production
* [biolabs/pubtrends-test](resources/docker/test/Dockerfile) - testing 

We use [Docker Hub](https://hub.docker.com/) to store built images. 

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`.\
   Ensure that file contains correct information about the database(s) (url, port, DB name, username and password).

2. Python environment `pubtrends` can be easily created using uv for launching Jupyter Notebook and Web Service:

    ```
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install -r pyproject.toml
    ```

3. Build base Docker image `biolabs/pubtrends` and nested image `biolabs/pubtrends-test` for testing.
    ```
    docker build -f resources/docker/main/Dockerfile -t biolabs/pubtrends --platform linux/amd64  .
    docker build  -f resources/docker/test/Dockerfile -t biolabs/pubtrends-test --platform linux/amd64 .
    ```

4. Init Postgres database.

    * Launch Docker image:
    ```
    docker run --rm --name pubtrends-postgres \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -v ~/postgres/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -p 5432:5432 \
        -d postgres:17
    ``` 
    * Create a database (once a database is created use `-d pubtrends` argument):
    ```
    psql -h localhost -p 5432 -U biolabs
    ALTER ROLE biolabs WITH LOGIN;
    CREATE DATABASE pubtrends OWNER biolabs;
    ```
    * Configure memory params in `~/postgres/pgdata/postgresql.conf`.
    ```
    # Memory settings
    effective_cache_size = 8GB  # ~ 50 to 75% (can be set precisely by referring to “top” free+cached)
    shared_buffers = 2GB        # ~ 1/4 – 1/3 total system RAM
    work_mem = 1GB            # For sorting, ordering etc
    max_connections = 4  # Total mem is work_mem * connections
    maintenance_work_mem = 1GB  # Memory for indexes, etc
    
    # Write performance
    checkpoint_timeout = 10min
    checkpoint_completion_target = 0.8
    synchronous_commit = off
    ```
   You can check current settings by command `SHOW ALL;` in psql console.

### Flask (Python) app configuration

The Flask components now use a class-based, environment-driven configuration.

- Loading: `app.config.from_object('pysrc.app.config.Config')`
- Environment selection via `APP_ENV` or `FLASK_ENV` (values: `development`, `testing`, `production`; default: `production`).
- Common environment variables:
  - `SECRET_KEY` — Flask secret key (autogenerated if not set)
  - `DATABASE_FILE` — SQLite filename used by admin service (default: `db.sqlite`)
  - `SQLALCHEMY_ECHO` — `1/true` to enable SQL echo
  - `SQLALCHEMY_TRACK_MODIFICATIONS` — default: `false`
  - `SECURITY_URL_PREFIX` — default: `/admin`
  - `SECURITY_PASSWORD_HASH` — default: `pbkdf2_sha512`
  - `SECURITY_PASSWORD_SALT` — required for stable password hashing
  - `SECURITY_LOGIN_URL` — default: `/login/`
  - `SECURITY_LOGOUT_URL` — default: `/logout/`
  - `SECURITY_POST_LOGIN_VIEW` — default: `/admin/`
  - `SECURITY_POST_LOGOUT_VIEW` — default: `/`

You can create a local `.env` file (see `.env.example`) to set these variables during development. If `python-dotenv` is installed, it will be auto-loaded.

## Kotlin/Java Build

Use the following command to test and build the JAR package:

   ```
   ./gradlew clean test shadowJar
   ```

## Papers downloading and processing

Postgresql should be configured and launched.

### Pubmed

Launch crawler to download and keep up to date a Pubmed database:

   ```
   java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase
   ``` 

Command line options supported:

* `resetDatabase` - clear current contents of the database (for development)
* `fillDatabase` - option to fill a database with Pubmed data. Can be interrupted at any moment.
* `lastId` - force downloading from given id from articles pack `pubmed20n{lastId+1}.xml`.

Updates - add the following line to crontab:

   ```
   crontab -e
   0 22 * * * java -cp pubtrends-<version>.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase | \
   tee -a crontab_update.log
   ```

### Semantic Scholar

Download Sample from [Semantic Scholar](https://www.semanticscholar.org/) or full archive. See Open Corpus.<br>
The latest release can be found at: https://api.semanticscholar.org/api-docs/datasets#tag/Release-Data

   ```
   curl https://api.semanticscholar.org/datasets/v1/release/
   ```

* Linux & Mac OS

   ```
   # Fail on errors
   set -euox pipefail 
  
   DATE="2022-05-01"
   PUBTRENDS_JAR=
  
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/manifest.txt
   echo "" > complete.txt
   N=$(cat manifest.txt | grep corpus | wc -l)
   cat manifest.txt | grep corpus | while read -r file; do 
      if [[ -z $(grep "$file" complete.txt) ]]; then
         echo "Processing $file / $N"
         wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/$file;
         java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase $(pwd)/$file
         rm $file;
         echo "$file" >> complete.txt
      fi;
   done
   java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --index --finish
   ```

* Windows 10 PowerShell

   ```
   $DATE = "2023-03-14
   $PUBTRENDS_JAR = 
   curl.exe -o .\manifest.txt https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/manifest.txt 
   echo "" > .\complete.txt
   foreach ($file in Get-Content .\manifest.txt) {
       $sel = Select-String -Path .\complete.txt -Pattern $file
       if ($sel -eq $null) {
          echo "Processing $file"
          curl.exe -o .\$file https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/$file
          java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase .\$file
          del ./$file
          echo $file >> .\complete.txt
       }
   }
   java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --index --finish
   ```

### Updating embeddings

Please ensure that embeddings Postgres DB with vector extension is up and running

   ```
   docker run --rm --name pgvector -p 5430:5432 \
        -m 32G \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -e POSTGRES_DB=pubtrends \
        -v ~/pgvector/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -d pgvector/pgvector:pg17
   ```

Then you'll be able to update embeddings with a commandline below. It will compute embeddings and store them into
the vector DB, and update the Faiss index for fast search.

   ```
   docker build -f pysrc/preprocess/embeddings/Dockerfile -t update_embeddings --platform linux/amd64 .
   docker run  -v ~/.pubtrends:/config:ro \
      -v ~/.pubtrends/logs:/logs \
      -v ~/.pubtrends/sentence-transformers:/sentence-transformers \
      -v ~/.pubtrends/nltk_data:/home/user/nltk_data \
      -v ~/.pubtrends/faiss:/faiss \
      -it update_embeddings /bin/bash
   
   uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install --no-cache sentence-transformers faiss-cpu
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   /bin/bash scripts/nlp.sh
   python pysrc/preprocess/update_embeddings.py
   ```

## Development

Please ensure that you have a database configured, up and running. \
Then launch web-service or use jupyter notebook for development.

### Web service

1. Create necessary folders with script `scripts/init.sh` and download prerequisites.
   ```
   bash scripts/init.sh
   bash scripts/nlp.sh
   ```

2. Start Redis
    ```
    docker run -p 6379:6379 redis:7.4.2
    ```

3. Configure Python environment with uv
    ```
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install -r pyproject.toml
    ```

4. Start Celery worker queue
    ```
    celery -A pysrc.celery.tasks worker -c 1 --loglevel=debug
    ```

5. Start flask server at http://localhost:5000/
    ```
    python -m pysrc.app.pubtrends_app
    ```

6. Start service for text embeddings based on either pretrained fasttext model or sentence-transformer
   at http://localhost:5001/
    ```
    python -m pysrc.endpoints.embeddings.fasttext.fasttext_app
    ```

or
```
python -m pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer_app
```

7. Optionally, start a semantic search service http://localhost:5002/
    ```
    python -m pysrc.semantic_search.semantic_search_app
    ```

### Jupyter notebook

Notebooks are located under the `/notebooks` folder. Please configure `PYTHONPATH` before using jupyter.

   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   jupyter notebook
   ```

## Testing

1. Start a Docker image with a Postgres environment for tests (Kotlin and Python development)
    ```
    docker run --rm --platform linux/amd64 --name pubtrends-test \
    --publish=5433:5432 --volume=$(pwd):/pubtrends -i -t biolabs/pubtrends-test
    ```

   NOTE: don't forget to stop the container afterward.

2. Kotlin tests

    ```
    ./gradlew clean test
    ```

3. Python tests with code style check for development (including integration with Kotlin DB writers)

    ```
    source .venv/bin/activate; pytest pysrc
    ```

4. Python tests within Docker (ensure that `./build/libs/pubtrends-dev.jar` file is present)

    ```
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

Please ensure that you have configured and prepared the database(s).

1. Modify file `config.properties` with information about the database(s). File from the project folder is used in this
   case.

2. Start Postgres server.

    ```
    docker run --rm --name pubtrends-postgres -p 5432:5432 \
        -m 32G \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -e POSTGRES_DB=pubtrends \
        -v ~/postgres/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -d postgres:17 
    ```
   NOTE: stop Postgres docker image with timeout `--time=300` to avoid DB recovery.\

   NOTE2: for speed reasons we use materialize views, which are updated upon successful database update. In case of
   an emergency stop, the view should be refreshed manually to ensure sort by citations works correctly:
    ```
    psql -h localhost -p 5432 -U biolabs -d pubtrends
    refresh materialized view matview_pmcitations;
    ``` 

3. Build ready for deployment package with script `scripts/dist.sh`.
   ```
   scripts/dist.sh build=build-number ga=google-analytics-id
   ```

4. Launch pubtrends with docker-compose (one of the options)
    ```
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
    ```
    # stop
    docker-compose -f docker-compose/semantic-search.yml down --remove-orphans
    # inpect logs
    docker-compose -f docker-compose/semantic-search.yml logs
    ```

   Pubtrends will be serving on port 5000.

## Maintenance

Use simple placeholder during maintenance.

   ```
   cd pysrc/app; python -m http.server 5000
   ```

## Release

* Update `CHANGES.md`
* Update version in `scripts/dist.sh`
* Launch `scripts/dist.sh`, `pubtrends-XXX.tar.gz` will be created in the `dist` directory.

# Authors

See [AUTHORS.md](AUTHORS.md) for a list of authors and contributors.

# Materials

* *Shpynov, O. and Kapralov, N., 2021, August. PubTrends: a scientific literature explorer. In Proceedings of the
  12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (pp.
  1-1).* https://doi.org/10.1145/3459930.3469501

* [Icons by Feather](https://feathericons.com/)

