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

# Configuration

1. Conda environment `pubtrends` can be easily created for launching Jupyter Notebook and Web Service:

    ```
    conda env create -f environment.yml
    conda activate pubtrends
    ```

2. Launch Postgres. 

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

3. Run `psql` to create a user and databases

   ```
   CREATE ROLE biolabs WITH PASSWORD 'password';
   ALTER ROLE biolabs WITH LOGIN;
   CREATE DATABASE pubtrends OWNER biolabs;
   CREATE DATABASE pubtrends_test OWNER biolabs;
   ```
   
3. Copy and modify `config.properties_examples` to `~/.pubtrends/config.properties`. 
Ensure that file contains correct information about the database (url, port, DB name, username and password).
 

## Papers crawling
 
### Pubmed

1. Use the following command to test and build the project:

   ```
   ./gradlew clean test shadowJar
   ```
     
2. Crawler is designed to download and keep up-to-date Pubmed database. Launch crawler:

   ```
   java -cp build/libs/crawler-dev.jar org.jetbrains.bio.pubtrends.pm.MainKt
   ``` 
   
3. Command line options supported:

   * `lastId` - in case of interruption use this parameter to restart the download from article pack `pubmed19n{lastId+1}.xml` 
   * `parserLimit` - maximum number of articles per XML to be parsed (useful for development)
   * `resetDatabase` - clear current contents of the database (useful for development)   

### Semantic Scholar

1. Download Sample from [Semantic Scholar](https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/sample-S2-records.gz)

2. Or full archive 
   ```
   cd <PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/manifest.txt
   wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/ -i manifest.txt
   ```
3. Unzip it (run in `<PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>`)
    ```
   gunzip -k *.gz
    ```

4. Add `<PATH_TO_SEMANTIC_SCHOLAR_ARCHIVE>` to `.pubtrends/config.properties`

5. Use the following command to test and build the project:

   ```
   ./gradlew clean test shadowJar
   ```
   
6. Launch Semantic Scholar Processing
    ```
    java -cp build/libs/crawler-dev.jar org.jetbrains.bio.pubtrends.ss.MainKt
    ```
## Service

### Jupyter Notebook
   ```
   jupyter notebook
   ```

### Web service
1. Start `Redis`
2. Start worker queue
    ```
    celery -A flask-async.celery worker -c 1 --loglevel=INFO
    ```
3. Start server
    ```
    python flask-async.py
    ```    
4. Open localhost:5000/

Useful links
------------
* Project presentation at [Computer Science Center](https://my.compscicenter.ru/media/projects/2019-spring/758/presentations/participants.pdf)
* JetBrains [TeamCity CI](https://teamcity.jetbrains.com/project.html?projectId=BioLabs)
* JetBrains Research BioLabs [homepage](https://research.jetbrains.org/groups/biolabs)