PubTrends
=========

A tool for analysis of trends & pivotal points in the scientific literature.

## Prerequisites

* JDK 8+
* Conda
* Python 3.6+
* Docker
* Neo4j 3.5+ with APOC 3.5.0.4 (in Docker)
* PostgreSQL 12 (in Docker)
* Redis 5.0 (in Docker)

## Configuration

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties`.\
Ensure that file contains correct information about the database(s) (url, port, DB name, username and password).

2. Conda environment `pubtrends` can be easily created for launching Jupyter Notebook and Web Service:

    ```
    conda env create -f environment.yml
    conda activate pubtrends
    ```

    Workaround for "image not found" error during neo4j pip install on newest Mac OS Sierra:
    `conda install -y conda-forge::ncurses`

3. Build `biolabs/pubtrends` Docker image (available on Docker hub).
    ```
    docker build -t biolabs/pubtrends .
    ```

4. Init Neo4j database and install APOC extension.
    * Prepare folders
    ```
    mkdir -p $HOME/neo4j/data $HOME/neo4j/logs $HOME/neo4j/plugins
    ```
   * Download the [latest release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/tag/3.5.0.4) of APOC
   * Place the binary JAR into your `$HOME/neo4j/plugins` folder
    * Launch Neo4j docker image.
    ```
    docker run --name pubtrends-neo4j \
        --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/var/lib/neo4j/data \
        --volume=$HOME/neo4j/logs:/logs \
        --volume=$HOME/neo4j/plugins:/plugins \
        -d neo4j:3.5
    ```   
   * Open Neo4j web browser to change default password (neo4j) to a strong one.

5. Init PostgreSQL database.
    
    * Launch Docker image:
    ```
    docker run --rm  --name pubtrends-postgres \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -v ~/postgres/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -p 5432:5432 \
        -d postgres:12
    ``` 
   * Create database:
    ```
    psql -h localhost -p 5432 -U biolabs
    ALTER ROLE biolabs WITH LOGIN;
    CREATE DATABASE pubtrends OWNER biolabs;
    ```
    * Configure `work_mem` to support search query sorted by citations in `postgresql.conf`.
      Experimentally, this amount is sufficient to search term 'computer' in Semantic Scholar sorted by citations count. 
    ```
    work_mem = '2048MB';   
    ```
   
## Kotlin/Java Build

Use the following command to test and build JAR package:

   ```
   ./gradlew clean test shadowJar
   ```

## Papers downloading and processing

Neo4j / PostgreSQL should be configured and launched.

### Pubmed

Launch crawler to download and keep up-to-date Pubmed database:

   ```
   java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase
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
         java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase $(pwd)/$file
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

1. Create necessary folders for logs and service database.
    ```
    mkdir ~/.pubtrends/logs
    mkdir ~/.pubtrends/database
    ```

2. Start Redis
    ```
    docker run redis:5.0
    ```
3. Start Celery worker queue
    ```
    celery -A pysrc.celery.tasks worker -c 1 --loglevel=debug
    ```
4. Start flask server at localhost:5000/
    ```
    python -m pysrc.app.app
    ```    
### Jupyter Notebook
   ```
   jupyter notebook
   ```


## Testing

1. Start Docker image with Neo4j / Postgres for tests (Kotlin and Python development)
    ```
    docker run --rm --name pubtrends-test \
    --publish=7474:7474 --publish=7687:7687 --publish=5432:5432 \
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
    ./gradlew shadowJar; conda activate pubtrends; pytest --flake8 pysrc
    ```

4. Tests within Docker (please ignore point 1)

    ```
    docker run --rm --volume=$(pwd):/pubtrends -t biolabs/pubtrends /bin/bash -c \
    "/usr/lib/postgresql/12/bin/pg_ctl -D /home/user/postgres start; sudo neo4j start; sleep 30s; \
    cd /pubtrends; mkdir ~/.pubtrends; cp config.properties ~/.pubtrends; \
    ./gradlew clean test; ./gradlew shadowJar; source activate pubtrends; pytest --flake8 pysrc"
    ```

## Deployment

Deployment is done with docker-compose. It is configured to start three containers:
* Gunicorn serving Flask app on HTTP port 80
* Redis as a message proxy
* Celery workers queue

Please ensure that you have configured and prepared the database(s).

1. Modify file `config.properties` with information about the database(s). 
   File from the project folder is used in this case.

2. Launch Neo4j database docker image (you can omit lines prefixed by `--env`).

    ```
    docker run --name pubtrends-neo4j \
        --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/var/lib/neo4j/data \
        --volume=$HOME/neo4j/logs:/logs \
        --volume=$HOME/neo4j/plugins:/plugins \
        -d neo4j:3.5
    ```
    
   See https://neo4j.com/developer/guide-performance-tuning/ for configuration details.
   
3. Start PostgreSQL server.

    ```
    docker run --rm  --name pubtrends-postgres -p 5432:5432 \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -e POSTGRES_DB=pubtrends \
        -v ~/postgres/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -d postgres:12 
    ```
   
4. Build ready for deployment package with script `dist.sh`.

5. Create logs folder within deployment package folder
   ```
   mkdir logs
   ```

6. Create necessary folders for logs and service database.
    ```
    mkdir ~/.pubtrends/logs
    mkdir ~/.pubtrends/database
    ```
 
7. Launch pubtrends with docker-compose.
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

8. During updates or other construction works consider launching simple reporter.
    ``` 
   docker run --rm -p 80:8000 -v $(pwd)/pysrc/app/construction:/construction \
        -t biolabs/pubtrends /bin/bash -c "python -m http.server -d /construction 8000"
   ```

# Authors

See [AUTHORS.md](AUTHORS.md) for a list of authors and contributors.

# Materials
* Project architecture [presentation](https://docs.google.com/presentation/d/131qvkEnzzmpx7-I0rz1om6TG7bMBtYwU9T1JNteRIEs/edit?usp=sharing) - summer 2019. 
* Review generation [presentation](https://my.compscicenter.ru/media/projects/2019-autumn/844/presentations/participants.pdf) - fall 2019.
* Extractive summarization [presentation](https://drive.google.com/file/d/1NnZ6JtJ2owtxFnuwKbARzOFM5_aHw6ls/view?usp=sharing) - spring 2020.
