PubTrends
=========

A tool for analysis of trends & pivotal points in the scientific literature.

## Prerequisites

* JDK 8+
* Conda
* Python 3.6+
* Docker

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`.\
Ensure that file contains correct information about the database (url, port, DB name, username and password).

2. Conda environment `pubtrends` can be easily created for launching Jupyter Notebook and Web Service:

    ```
    conda env create -f environment.yml
    conda activate pubtrends
    ```

3. Build `biolabs/pubtrends` Docker image (available on Docker hub).
    ```
    docker build -t biolabs/pubtrends .
    ```

4. Launch Neo4j and PostgreSQL dev docker image.
    ```
    docker run --rm --name pubtrends-docker \
    --publish=5433:5432 --publish=7474:7474 --publish=7687:7687 \
    --volume=$(pwd):/pubtrends -d -t biolabs/pubtrends
    ```
   
## Build

Use the following command to test and build the project:

   ```
   ./gradlew clean test shadowJar
   ```

## Papers processing
 
### Pubmed

Launch crawler to download and keep up-to-date Pubmed database:

   ```
   java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.MainKt
   ``` 
   
   Command line options supported:
   * `lastId` - in case of interruption use this parameter to restart the download from article pack `pubmed19n{lastId+1}.xml` 
   * `resetDatabase` - clear current contents of the database (useful for development)   

### Semantic Scholar

1. Add `<PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>` to `.pubtrends/config.properties`     

2. Download Sample from [Semantic Scholar](https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/sample-S2-records.gz)
   Or full archive 
   ```
   cd <PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/manifest.txt
   cat manifest.txt | grep corpus | while read -r url; do 
      wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$url;
      java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.ss.MainKt --fillDatabase
      rm $(echo $url | sed -E 's#^.*/##g');
   done
   ```

3. Build Semantic Scholar Indexes
    ```
    java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.ss.MainKt --createIndex
    ```
   
   Additional command line options supported:

   * `resetDatabase` - clear current contents of the database (useful for development) 
   * `fillDatabase` - create and fill database with Semantic Scholar data
   * `createIndex` - create index for already created tables
   
## Service

Several front-ends are supported.

### Jupyter Notebook
   ```
   jupyter notebook
   ```

### Web service

1. Start Redis

2. Start Celery worker queue
    ```
    celery -A models.celery.tasks worker -c 1 --loglevel=info
    ```
3. Start flask server at localhost:5000/
    ```
    python models/flask-app.py
    ```    

### Deployment

Launch Gunicorn serving Flask app on HTTP port 80, Redis and Celery in containers by the command:

1. Launch Neo4j docker image.
    ```
    docker run --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/data --volume=$HOME/neo4j/logs:/logs neo4j:3.5
    ```

2. Launch docker-compose config
    ```
    # start
    docker-compose up -d --build
    # stop
    docker-compose down
    # inpect logs
    docker-compose logs
    ```

## Testing

1. Start Docker image with Postgres and Neo4j for tests

    ```
    docker run --rm --name pubtrends-docker \
    --publish=5433:5432 --publish=7474:7474 --publish=7687:7687 \
    --volume=$(pwd):/pubtrends -d -t biolabs/pubtrends
    ```

    Check access to Postgresql: `psql postgresql://biolabs:password@localhost:5433/pubtrends_test`
    Check access to Neo4j web browser: `http://localhost:7474`

2. Kotlin tests

    ```
    ./gradlew clean test
    ```

3. Python tests with codestyle check
    
    ```
    source activate pubtrends; python -m pytest --codestyle models
    ```

4. Python tests with codestyle check within Docker

    ```
    docker run --rm --volume=$(pwd):/pubtrends -t biolabs/pubtrends /bin/bash -c \
    "/usr/lib/postgresql/11/bin/pg_ctl -D /home/user/postgres start; sudo neo4j start; sleep 10s; \
    source activate pubtrends; cd /pubtrends; python -m pytest --codestyle models;"
    ```
   
# Authors

See [AUTHORS.md](AUTHORS.md) for a list of authors and contributors.

# Materials
* Project architecture [presentation](https://docs.google.com/presentation/d/131qvkEnzzmpx7-I0rz1om6TG7bMBtYwU9T1JNteRIEs/edit?usp=sharing) - summer 2019. 
* Review generation [presentation](https://my.compscicenter.ru/media/projects/2019-autumn/844/presentations/participants.pdf) - fall 2019.

