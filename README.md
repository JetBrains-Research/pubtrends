[![JetBrains team project](https://jb.gg/badges/team.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)


PubTrends
=========

A tool for analysis of trends & pivotal points in the scientific literature.

## Prerequisites

* JDK 8+
* PostgreSQL 10.5+
* Conda
* Python 3.6+
* Celery
* Redis
* Docker

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`. 
Ensure that file contains correct information about the database (url, port, DB name, username and password).

2. Conda environment `pubtrends` can be easily created for launching Jupyter Notebook and Web Service:

    ```
    conda env create -f environment.yml
    conda activate pubtrends
    ```

3. Launch Postgres. 

    Mac OS
    ```
    # start
    pg_ctl -D /usr/local/var/postgres -l /usr/local/var/postgres/server.log start
    # stop
    pg_ctl -D /usr/local/var/postgres stop -s -m fast
    ```
    Ubuntu
    ```
    # start
    service postgresql start
    # start
    service postgresql stop 
    ```

4. Run `psql` to create a user and databases

   ```
   CREATE ROLE biolabs WITH PASSWORD 'password';
   ALTER ROLE biolabs WITH LOGIN;
   CREATE DATABASE pubtrends OWNER biolabs;
   ```
   Create testing database if you don't want to use Docker based Postgresql for tests
   ```
   CREATE DATABASE pubtrends_test OWNER biolabs;
   ```

## Papers processing
 
### Pubmed

1. Use the following command to test and build the project:

   ```
   ./gradlew clean test shadowJar
   ```
     
2. Crawler is designed to download and keep up-to-date Pubmed database. Launch crawler:

   ```
   java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.MainKt
   ``` 
   
   Command line options supported:
   * `lastId` - in case of interruption use this parameter to restart the download from article pack `pubmed19n{lastId+1}.xml` 
   * `resetDatabase` - clear current contents of the database (useful for development)   

### Semantic Scholar

1. Download Sample from [Semantic Scholar](https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/sample-S2-records.gz)
   Or full archive 
   ```
   cd <PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/manifest.txt
   wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/ -i manifest.txt
   ```
2. Add `<PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>` to `.pubtrends/config.properties`

3. Use the following command to test and build the project:

   ```
   ./gradlew clean test shadowJar
   ```
   
4. Launch Semantic Scholar Processing
    ```
    java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.ss.MainKt
    ```
   Command line options supported:

   * `resetDatabase` - clear current contents of the database (useful for development) 
   * `fillDatabase` - create and fill database with Semantic Scholar data
   * `createIndex` - create index for already created tables
   
   For example, if you launch Semantic Scholar Processing for the first time, 
   you need to use `fillDatabase` and `createIndex` options. 

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
    celery -A models.celery.tasks worker --loglevel=info
    ```
3. Start flask server at localhost:5000/
    ```
    python models/flask-app.py
    ```    

### Deployment with Docker Compose

Launch Gunicorn serving Flask app, Redis and Celery in containers by the command:
    
    ```
    # start
    docker-compose up -d --build
    # stop
    docker-compose down
    # inpect logs
    docker-compose logs
    ```

## Testing

### Docker Image for testing

1. Build `biolabs/pubtrends` Docker image
    ```
    docker build -t biolabs/pubtrends .
    ```

2. Start Docker image for Kotlin tests
    ```
    docker run --name pg-docker -p 5433:5432 -v $(pwd):/pubtrends:ro -d biolabs/pubtrends
    ```

    Check access:
    ```
    psql postgresql://biolabs:password@localhost:5433/pubtrends_test
    ```

3. Kotlin tests

    ```
    ./gradlew clean test
    ```

4. Python tests

    ```
    docker run -v $(pwd):/pubtrends:ro -t biolabs/pubtrends /bin/bash \
    -c "/usr/lib/postgresql/10/bin/pg_ctl -D /home/user/postgres start; \
    source activate pubtrends; cd /pubtrends; python -m pytest models/test"
    ```
   
5. Python code style tests
    ```
    docker run -v $(pwd):/pubtrends:ro -t biolabs/pubtrends /bin/bash \
    -c "source activate pubtrends; cd /pubtrends; python -m pytest --codestyle -m codestyle"
    ```


# Materials

* Project presentation at [Computer Science Center](https://my.compscicenter.ru/media/projects/2019-spring/758/presentations/participants.pdf)
* Project [workflow](https://docs.google.com/presentation/d/1rIVxEmpJhQWfFXsIWMwg9vZsKSTDEv7Whxe39EuJn60/edit#slide=id.g5efdb45a3b_0_30)
* JetBrains Research BioLabs [homepage](https://research.jetbrains.org/groups/biolabs)
