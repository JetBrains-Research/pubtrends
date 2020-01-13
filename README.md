PubTrends
=========

A tool for analysis of trends & pivotal points in the scientific literature.

## Prerequisites

* JDK 8+
* Conda
* Python 3.6+
* Docker
* Neo4j 3.5+ with APOC 3.5.0.4 (Optional, can be used in Docker)
* Redis 5.0 (Optional, can be used in Docker)

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`.\
Ensure that file contains correct information about the database(s) (url, port, DB name, username and password).

2. Conda environment `pubtrends` can be easily created for launching Jupyter Notebook and Web Service:

    ```
    conda env create -f environment.yml
    conda activate pubtrends
    ```

3. Build `biolabs/pubtrends` Docker image (available on Docker hub).
    ```
    docker build -t biolabs/pubtrends .
    ```

4. Configure Neo4j and install APOC extension.
    * Prepare folders
    ```
    mkdir -p $HOME/neo4j/data $HOME/neo4j/conf $HOME/neo4j/logs $HOME/neo4j/plugins
    ```
    * Launch Neo4j docker image to create config file.
    ```
    docker run --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/var/lib/neo4j/data \
        --volume=$HOME/neo4j/conf:/var/lib/neo4j/conf \
        --volume=$HOME/neo4j/logs:/logs \
        --volume=$HOME/neo4j/plugins:/plugins \
        neo4j:3.5
    ```
   * Download the [latest release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/tag/3.5.0.4) of APOC
   * Place the binary JAR into your `$HOME/neo4j/plugins` folder
   * Add line `dbms.security.procedures.unrestricted=apoc.*` to the end of config file
    `$HOME/neo4j/conf/neo4j.conf` to allow usage of the procedures.
   * Open Neo4j web browser to change default password (neo4j) to a strong one.


## Kotlin/Java Build

Use the following command to test and build JAR package:

   ```
   ./gradlew clean test shadowJar
   ```

## Papers downloading and processing

Neo4j and/or PostgreSQL should be configured and launched.
You can always inspect data structure in Neo4j web browser.

### Pubmed

Launch crawler to download and keep up-to-date Pubmed database:

   ```
   java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.MainKt --fillDatabase
   ``` 
   
   Command line options supported:
   * `resetDatabase` - clear current contents of the database (for development)
   * `fillDatabase` - option to fill database with Pubmed data. Can be interrupted at any moment.
   * `lastId` - force downloading from given id from articles pack `pubmed20n{lastId+1}.xml`. 
   

### Optional: Semantic Scholar


Download Sample from [Semantic Scholar](https://www.semanticscholar.org/) or full archive. 
   ```
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-01-01/manifest.txt
   echo "" > complete.txt
   cat manifest.txt | grep corpus | while read -r file; do 
      if [[ -z $(grep "$file" complete.txt) ]]; then
         wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-01-01/$file;
         java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.ss.MainKt --fillDatabase $(pwd)/$file
         rm $file;
         echo "$file" >> complete.txt
      fi;
   done
   ```

   Command line options supported:
   * `resetDatabase` - clear current contents of the database (useful for development) 
   * `fillDatabase` - create and fill database with Semantic Scholar data from file


## Development

Several front-ends are supported.
Please ensure that you have Database configured, up and running.

### Web service

1. Start Redis
    ```
    docker run redis:5.0
    ```
2. Start Celery worker queue
    ```
    celery -A models.celery.tasks worker -c 1 --loglevel=debug
    ```
3. Start flask server at localhost:5000/
    ```
    python models/flask-app.py
    ```    
### Jupyter Notebook
   ```
   jupyter notebook
   ```


## Testing

1. Start Docker image with Postgres and Neo4j for tests (Kotlin and Python tests development)
    ```
    docker run --rm --name pubtrends-docker \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$(pwd):/pubtrends -d -t biolabs/pubtrends
    ```

    Check access to Neo4j web browser: `http://localhost:7474`
    NOTE: please don't forget to stop the container afterwards.

2. Kotlin tests

    ```
    ./gradlew clean test
    ```

3. Python tests with codestyle check for development
    
    ```
    source activate pubtrends; python -m pytest --pycodestyle models
    ```

4. Python tests with codestyle check within Docker (please ignore point 1)

    ```
    docker run --rm --volume=$(pwd):/pubtrends -t biolabs/pubtrends /bin/bash -c \
    "sudo neo4j start; sleep 20s; \
    cd /pubtrends; cp config.properties ~/.pubtrends; \
    source activate pubtrends; python -m pytest --pycodestyle models"
    ```

## Deployment

Deployment is done with docker-compose. It is configured to start three containers:
* Gunicorn serving Flask app on HTTP port 80
* Redis as a message proxy
* Celery workers queue

Please ensure that you have configured and prepared the database(s).

1. Modify file `config.properties` with information about the database(s).

2. Launch Neo4j database docker image.
    ```
    docker run --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/var/lib/neo4j/data \
        --volume=$HOME/neo4j/conf:/var/lib/neo4j/conf \
        --volume=$HOME/neo4j/logs:/logs \
        --volume=$HOME/neo4j/plugins:/plugins \
        neo4j:3.5
    ```
3. Build ready for deployment package with script `dist.sh`.

4. Launch pubtrends with docker-compose.
    ```
    # start
    docker-compose up -d --build
    ```
    Use these commands to stop compose build and check logs:
    ```
    # stop
    docker-compose down
    # inpect logs
    docker-compose logs
    ```


# Authors

See [AUTHORS.md](AUTHORS.md) for a list of authors and contributors.

# Materials
* Project architecture [presentation](https://docs.google.com/presentation/d/131qvkEnzzmpx7-I0rz1om6TG7bMBtYwU9T1JNteRIEs/edit?usp=sharing) - summer 2019. 
* Review generation [presentation](https://my.compscicenter.ru/media/projects/2019-autumn/844/presentations/participants.pdf) - fall 2019.
