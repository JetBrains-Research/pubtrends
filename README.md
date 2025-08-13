[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Build Status](http://teamcity.jetbrains.com/app/rest/builds/buildType:(id:BioLabs_Pubtrends_Deployment)/statusIcon.svg)](http://teamcity.jetbrains.com/viewType.html?buildTypeId=BioLabs_Pubtrends_Deployment&guest=1)
[![DOI](https://zenodo.org/badge/151591143.svg)](https://doi.org/10.5281/zenodo.15131474)

PubTrends
=========

PubTrends is a scientific literature exploratory tool for analyzing topics of a research field and similar papers
analysis. 

Available online at: https://pubtrends.info/

With PubTrends, you can:
* Gain a concise overview of your research area.
* Explore popular trends and impactful publications.
* Discover new and promising research directions.

Datasets:
* [Pubmed](https://pubmed.ncbi.nlm.nih.gov) 30 mln papers and 175 mln citations
* [Semantic Scholar](https://www.semanticscholar.org) 170 mln papers and 600 mln citations

![Scheme](pysrc/app/static/about_pubtrends_scheme.png?raw=true "Title")

## Technical details

PubTrends is a web service developed in Python and JavaScript, designed to analyze and visualize information about scientific publications. It uses PostgreSQL as its main database for storing details like titles, abstracts, authors, and citations, along with its built-in text search for full-text search functionality. A Kotlin ORM is used to manage the database, while a separate SQLite database stores user roles and admin credentials.

The web service is powered by Flask and Gunicorn, with Celery managing asynchronous tasks and Redis acting as the message broker. For data manipulation and analysis, libraries such as Pandas, NumPy, and Scikit-Learn are used, while text processing relies on NLTK and SpaCy. Graphs and embeddings are handled using NetworkX, word2vec (via GenSim), and a custom node2vec implementation.

For data visualization, Bokeh, Holoviews, Seaborn, and Matplotlib are used, with Bokeh providing interactive plots for web pages and Jupyter notebooks. The frontend is built with Bootstrap for layout, jQuery for interactivity, and Cytoscape.js for rendering graphs. 

Please refer to [environment.yml](environment.yml) for the full list of libraries used in the project.

### Docker

Two Docker images are used for testing and deployment: [biolabs/pubtrends-test](resources/docker/main/Dockerfile)
and [biolabs/pubtrends](resources/docker/test/Dockerfile), respectively. We use [Docker Hub](https://hub.docker.com/) to
store built images. Service deployment is done with Docker compose, which launches Redis container, Celery container and
Gunicorn container.

Please refer to [docker-compose.yml](docker-compose.yml) for more information about deployment.

### Testing and CI

Testing is done with Pytest and JUnit. Flake8 linter is used for quality assessment of Python code. Python tests are
launched within Docker. Continuous integration is done with TeamCity using build chains.

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`.\
   Ensure that file contains correct information about the database(s) (url, port, DB name, username and password).

2. Conda environment `pubtrends` can be easily created for launching Jupyter Notebook and Web Service:

    ```
    conda env create -f environment.yml
    source activate pubtrends
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
   python pysrc/preprocess/update_embeddings.py --host localhost --port 5430 --database pubtrends --user biolabs --password mysecretpassword --max-year <MAX_YEAR> --min-year <MIN_YEAR>
   ```


## Development

Please ensure that you have a database configured, up and running. \
Then launch web-service or use jupyter notebook for development.

### Web service

1. Create necessary folders with script `init.sh` and download prerequisites.
   ```
   source activate pubtrends \
      && python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng \
      punkt punkt_tab stopwords wordnet omw-1.4 \
      && python -m spacy download en_core_web_sm
   ```

2. Start Redis
    ```
    docker run -p 6379:6379 redis:7.4.2
    ```

3. Configure conda environment `pubtrends`
    ```
    conda env create -f environment.yml
    ```
   Enable environment by command `source activate pubtrends`.

4. Start Celery worker queue
    ```
    celery -A pysrc.celery.tasks worker -c 1 --loglevel=debug
    ```

5. Start flask server at http://localhost:5000/
    ```
    python -m pysrc.app.pubtrends_app
    ```

6. Start service for text embeddings based on either pretrained fasttext model or sentence-transformer at http://localhost:5001/
    ```
    python -m pysrc.endpoints.embeddings.fasttext.fasttext_app
    ```
or 
    ```
    python -m pysrc.endpoints.embeddings.sentence_transformer.sentence_transformer_app
    ```
    
7. Optionally start semantic search service http://localhost:5002/
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
    source activate pubtrends; pytest pysrc
    ```

4. Python tests within Docker (ensure that `./build/libs/pubtrends-dev.jar` file is present)

    ```
    docker run --rm --platform linux/amd64 --volume=$(pwd):/pubtrends -t biolabs/pubtrends-test /bin/bash -c \
    "/usr/lib/postgresql/17/bin/pg_ctl -D /home/user/postgres start; \
    cd /pubtrends; cp config.properties /home/user/.pubtrends/; \
    source activate pubtrends; pytest pysrc"
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

3. Build ready for deployment package with script `dist.sh`.
   ```
   dist.sh build=build-number ga=google-analytics-id
   ```

4. Launch pubtrends with docker-compose (one of the options)
    ```
    # start with local word2vec tf-idf tokens embeddings
    docker-compose -f docker-compose.yml up --build
    
    # start with BioWord2Vec tokens embeddings
    docker-compose -f docker-compose.fasttext.yml up --build
    
    # start with Sentence Transformer for text embeddings
    docker-compose -f docker-compose.sentence-transformer.yml up --build
    
    # Start with Semantic Search based on Sentence Transformer
    docker-compose -f docker-compose.semantic-search.yml up --build 
    ```
   Use these commands to stop compose build and check logs:
    ```
    # stop
    docker-compose down
    # inpect logs
    docker-compose logs
    ```

   Pubtrends will be serving on port 5000.

## Maintenance

Use simple placeholder during maintenance.

   ```
   cd pysrc/app; python -m http.server 5000
   ```

## Release

* Update `CHANGES.md`
* Update version in `dist.sh`
* Launch `dist.sh`, `pubtrends-XXX.tar.gz` will be created in the `dist` directory.

# Authors

See [AUTHORS.md](AUTHORS.md) for a list of authors and contributors.

# Materials

* *Shpynov, O. and Kapralov, N., 2021, August. PubTrends: a scientific literature explorer. In Proceedings of the
12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (pp. 1-1).* https://doi.org/10.1145/3459930.3469501

* [Icons by Feather](https://feathericons.com/)

