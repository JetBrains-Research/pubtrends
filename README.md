[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Build Status](http://teamcity.jetbrains.com/app/rest/builds/buildType:(id:BioLabs_Pubtrends_Deployment)/statusIcon.svg)](http://teamcity.jetbrains.com/viewType.html?buildTypeId=BioLabs_Pubtrends_Deployment&guest=1)


PubTrends
=========

PubTrends is a scientific literature exploratory tool for analyzing topics of a research field and similar papers
analysis. It runs a Pubmed or Semantic Scholar search and allows user to explore high-level structure of result papers.

Open Access Paper: [https://doi.org/10.1145/3459930.3469501](https://doi.org/10.1145/3459930.3469501), poster
is [here](https://drive.google.com/file/d/1SeqJtJtaHSO6YihG2905boOEYL1NiSP1/view?usp=sharing). \
*Citation: Shpynov, O. and Nikolai, K., 2021, August. PubTrends: a scientific literature explorer. In Proceedings of the
12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (pp. 1-1).*

![Scheme](pysrc/app/static/about_pubtrends_scheme.png?raw=true "Title")

## Technical details

PubTrends is a web service, written in Python and Javascript. It uses PostgreSQL to store information about scientific
publications.

### Libraries

Web service is built with Gunicorn and Flask. Asynchronous computations are supported with Celery tasks queue and Redis
as message broker. We use PostgreSQL to store information about papers: titles, abstracts, authors and citations
information. PostgreSQL built-in text search engine is used for full text search. Kotlin PostgreSQL ORM is used to
store papers in the database. MySQL database is used to store technical user information including users roles and admin
credentials for admin dashboard.

All the data manipulations are made with Pandas, Numpy and Scikit-Learn libraries. The service uses Python Nltk and
Spacy libraries for text processing and analysis. Graph objects are processed with NetworkX library, papers embeddings
are created with word2vec library from GenSim and in-house node2vec implementation based on word2vec. All the plots are
created with Bokeh, Holoviews, Seaborn and Matplotlib libraries. Interactive Bokeh plots are used in web pages and
Jupyter notebook experiments. Frontend is created with Bootstrap, JQuery and Cytoscape-JS for graphs rendering.

Please refer to [environment.yml](environment.yml) for the full list of libraries used in the project.

### Docker

Two Docker images are used for testing and deployment: [biolabs/pubtrends-test](resources/docker/main/Dockerfile)
and [biolabs/pubtrends](resources/docker/test/Dockerfile), respectively. We use [Docker Hub](https://hub.docker.com/) to
store built images. Service deployment is done with Docker compose, which launches Redis container, Celery container and
Gunicorn container.

Please refer to [docker-compose.yml](docker-compose.yml) for more information about deployment.

### Testing and CI

Testing is done with Pytest and JUnit. Flake8 linter is used for quality assessment of Python code. Python tests are
launched within Docker. Continuous integration is done with TeamCity using build chains. Each commit is assigned with
build number which is later used for *Python tests*, *Kotlin tests* and *Docker-compose tests* build configurations.
Dedicated *Distributive* build configuration is used to build Kotlin code and pack all the Python code into distribution
packages, this configuration depends on tests configuration and is being executed only when all the tests are passed
successfully. Distribution packages are used for database updates and web service deployment.

### Product metrics

We use Google Analytics to collect information about site visitors as well as internal statistics collector available
from `/admin` dashboard, which shows requests and results, execution times and feedback from users.

## Development Prerequisites

* JDK 8+
* Conda
* Python 3.6+
* Docker
* PostgreSQL 14 (in Docker)
* Redis 5.0 (in Docker)

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
    docker build -f resources/docker/main/Dockerfile -t biolabs/pubtrends .
    docker build -f resources/docker/test/Dockerfile -t biolabs/pubtrends-test . 
    ```

4. Init PostgreSQL database.

    * Launch Docker image:
    ```
    docker run --rm  --name pubtrends-postgres \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -v ~/postgres/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -p 5432:5432 \
        -d postgres:14
    ``` 
    * Create database:
    ```
    psql -h localhost -p 5432 -U biolabs
    ALTER ROLE biolabs WITH LOGIN;
    CREATE DATABASE pubtrends OWNER biolabs;
    ```
    * Configure memory params in `~/postgres/pgdata/postgresql.conf`.
    ```
    # Memory settings
    effective_cache_size = 4GB  # ~ 50 to 75% (can be set precisely by referring to “top” free+cached)
    shared_buffers = 2GB        # ~ 1/4 – 1/3 total system RAM
    work_mem = 512MB            # For sorting, ordering etc
    max_client_connections = 4  # Total mem is work_mem * connections
    maintenance_work_mem = 1GB  # Memory for indexes, etc
    
    # Write performance
    checkpoint_timeout = 10min
    checkpoint_completion_target = 0.8
    synchronous_commit = off
    ```

5. Clone the [JetBrains-Research/pubtrends-review](https://github.com/JetBrains-Research/pubtrends-review) repository to
   the working directory, and enable it in `~/.pubtrends/config.properties` file.

   ```
   git clone git@github.com:JetBrains-Research/pubtrends-review.git
   ```

## Kotlin/Java Build

Use the following command to test and build JAR package:

   ```
   ./gradlew clean test shadowJar
   ```

## Papers downloading and processing

PostgreSQL should be configured and launched.

### Pubmed

Launch crawler to download and keep up-to-date Pubmed database:

   ```
   java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase
   ``` 

Command line options supported:

* `resetDatabase` - clear current contents of the database (for development)
* `fillDatabase` - option to fill database with Pubmed data. Can be interrupted at any moment.
* `lastId` - force downloading from given id from articles pack `pubmed20n{lastId+1}.xml`.

Updates - add the following line to crontab:
```
crontab -e
0 22 * * * java -cp pubtrends-<version>.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase | \
tee -a crontab_update.log
```

### Semantic Scholar

Download Sample from [Semantic Scholar](https://www.semanticscholar.org/) or full archive. See Open Corpus.<br>
Instructions are for the corpus 2021-11-01. \
Replace `<PATH_TO_PUBTRENDS.JAR>` with actual path to Jar file.

* Linux & Mac OS

   ```
   # Fail on errors
   set -euox pipefail 
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2021-11-01/manifest.txt
   echo "" > complete.txt
   cat manifest.txt | grep corpus | while read -r file; do 
      if [[ -z $(grep "$file" complete.txt) ]]; then
         echo "Processing $file"
         wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2021-11-01/$file;
         java -cp <PUBTRENDS.JAR> org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase $(pwd)/$file
         rm $file;
         echo "$file" >> complete.txt
      fi;
   done
   java -cp <PUBTRENDS.JAR> org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --finish
   ```

* Windows 10 PowerShell

   ```
   curl.exe -o .\manifest.txt https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2021-11-01/manifest.txt 
   echo "" > .\complete.txt
   foreach ($file in Get-Content .\manifest.txt) {
       $sel = Select-String -Path .\complete.txt -Pattern $file
       if ($sel -eq $null) {
          echo "Processing $file"
          curl.exe -o .\$file https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2021-11-01/$file
          java -cp <PUBTRENDS.JAR> org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase .\$file
          del ./$file
          echo $file >> .\complete.txt
       }
   }
   java -cp <PUBTRENDS.JAR> org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --finish
   ```

## Development

Please ensure that you have database configured, up and running. \
Then launch web-service or use jupyter notebook for development.

### Web service

1. Create necessary folders with script `init.sh`.

2. Start Redis
    ```
    docker run -p 6379:6379 redis:5.0
    ```

3. Configure conda environment `pubtrends`
    ```
    conda env create -f environment.yml
    ```
   Enable environment by command `source activate pubtrends`.

4. Download nltk & spacy resources
    ```
    source activate pubtrends
    python -m nltk.downloader averaged_perceptron_tagger punkt stopwords wordnet
    python -m spacy download en_core_web_sm
    ```

5. Start Celery worker queue
    ```
    celery -A pysrc.celery.tasks worker -c 1 --loglevel=debug
    ```

6. Start flask server at http://localhost:5000/
    ```
    python -m pysrc.app.app
    ```    

7. Start service for text embeddings based on pretrained fasttext model at http://localhost:8081/
    ```
    python -m pysrc.fasttext.fasttext_app
    ```    

### Jupyter notebook

Notebooks are located under the `/notebooks` folder. Please configure `PYTHONPATH` before using jupyter.

   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   jupyter notebook
   ```

## Testing

1. Start Docker image with PostgreSQL environment for tests (Kotlin and Python development)
    ```
    docker run --rm --name pubtrends-test \
    --publish=5432:5432 --volume=$(pwd):/pubtrends -i -t biolabs/pubtrends-test
    ```

   NOTE: don't forget to stop the container afterwards.

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
    docker run --rm --volume=$(pwd):/pubtrends -t biolabs/pubtrends-test /bin/bash -c \
    "/usr/lib/postgresql/14/bin/pg_ctl -D /home/user/postgres start; \
    cd /pubtrends; mkdir ~/.pubtrends; cp config.properties ~/.pubtrends; \
    source activate pubtrends; pytest pysrc"
    ```

## Deployment

Deployment is done with docker-compose. It is configured to start the following containers:

* Gunicorn serving main pubtrends Flask app
* Redis as a message proxy
* Celery workers queue
* Gunicorn fasttext model app

Please ensure that you have configured and prepared the database(s).

1. Modify file `config.properties` with information about the database(s). File from the project folder is used in this
   case.

2. Start PostgreSQL server.

    ```
    docker run --rm  --name pubtrends-postgres -p 5432:5432 \
        --shm-size=8g \
        -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
        -e POSTGRES_DB=pubtrends \
        -v ~/postgres/:/var/lib/postgresql/data \
        -e PGDATA=/var/lib/postgresql/data/pgdata \
        -d postgres:14 
    ```
   NOTE: stop PostgreSQL docker image with timeout `--time=300` to avoid DB recovery.\

   NOTE2: for speed reason we use materialize views, which are updated upon successful database update. In case of
   emergency stop, the view should be refreshed manually to ensure sort by citations works correctly:
    ```
    psql -h localhost -p 5432 -U biolabs -d pubtrends
    refresh materialized view matview_pmcitations;
    ``` 

3. Build ready for deployment package with script `dist.sh`.
   ```
   dist.sh build=build-number ga=google-analytics-id
   ```


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

   Pubtrends will be serving on port 8888.
5. Nginx is used to proxy all traffic to port 8888 and redirect http -> https with Let's encrypt certificates.

## Maintenance

   Use simple placeholder during maintenance.
   ```
   cd pysrc/app; python -m http.server 8888
   ```


## Release

* Update `CHANGES.txt`
* Update version in `dist.sh`
* Launch `dist.sh`, `pubtrends-XXX.tar.gz` will be created in the `dist` directory.

# Authors

See [AUTHORS.md](AUTHORS.md) for a list of authors and contributors.

# Materials

* Open Access Paper: [https://doi.org/10.1145/3459930.3469501](https://doi.org/10.1145/3459930.3469501), poster
  [here](https://drive.google.com/file/d/1SeqJtJtaHSO6YihG2905boOEYL1NiSP1/view?usp=sharing) - 2021.
* Project overview
  [presentation](https://docs.google.com/presentation/d/131qvkEnzzmpx7-I0rz1om6TG7bMBtYwU9T1JNteRIEs/edit?usp=sharing) -
  summer 2019.
* Review generation
  [presentation](https://my.compscicenter.ru/media/projects/2019-autumn/844/presentations/participants.pdf) - fall 2019.
* Extractive summarization
  [presentation](https://drive.google.com/file/d/1NnZ6JtJ2owtxFnuwKbARzOFM5_aHw6ls/view?usp=sharing) - spring 2020.
* Paper ["Automatic generation of reviews of scientific papers"](https://arxiv.org/abs/2010.04147) - 2021.
* [Icons by Feather](https://feathericons.com/)

