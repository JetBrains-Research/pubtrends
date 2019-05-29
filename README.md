# PubTrends

A tool for analysis of trends & pivotal points in the scientific literature.

Project workflow: https://drive.google.com/open?id=17oQAuMJ0vmDgzucGxJT_xgQBCkLEio3HIeG_sfgMtcc

## Prerequisites

* JDK 8
* PostgreSQL (tested with 10.5)

## Setup

1. Clone the repository.

2. Launch Postgres. Commands are for Mac OS.
    ```
    # stop
    pg_ctl -D /usr/local/var/postgres stop -s -m fast
    # start
    pg_ctl -D /usr/local/var/postgres -l /usr/local/var/postgres/server.log start
    ```

2. Run `psql` to create a user and a database in PostgreSQL:

   ```
   CREATE ROLE biolabs WITH PASSWORD 'pubtrends';
   ALTER ROLE "biolabs" WITH LOGIN;
   CREATE DATABASE pubmed OWNER biolabs; 
   ```
   
3. Copy and modify `config.properties_examples` to `~/.pubtrends/config.properties`. 
Ensure that file contains correct information about the database (url, port, DB name, username and password).
   
   ```
   url = localhost
   port = 5432
   database = pubmed
   username = biolabs
   password = pubtrends
   collectStats = true
   ```
`collectStats` option is used to count number of XML tag occurrences (useful for development)

## Startup

1. Command line options supported:

   * `lastId` - in case of interruption use this parameter to restart the download from article pack `pubmed19n{lastId+1}.xml` 
   * `parserLimit` - maximum number of articles per XML to be parsed (useful for development)
   * `resetDatabase` - clear current contents of the database (useful for development)
   
2. Use the following command to test and build the project:

   ```
   ./gradlew clean test shadowJar
   ```
     
3. Crawler is designed to download and keep up-to-date PubMed database. Launch crawler:

   ```
   java -jar build/libs/crawler-dev.jar
   ``` 

## Jupyter Notebook frontend
   ```
   jupyter notebook
   ```

## Web service
1. Install and start `Redis`.
2. start server
    ```
    python flask-async.py
    ```    
3. start worker
    ```
    celery -A flask-async.celery worker -c 1 --loglevel=DEBUG
    ```
4. Open localhost:5000/
