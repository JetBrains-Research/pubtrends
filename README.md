# PubTrends

A tool for analysis of trends & pivotal points in the scientific literature.

Project workflow: https://drive.google.com/open?id=17oQAuMJ0vmDgzucGxJT_xgQBCkLEio3HIeG_sfgMtcc

### Prerequisites

* PostgreSQL (tested with 10.5)

### Getting started

1. Clone the repository.

2. Run ```psql``` to create a user and a database in PostgreSQL:

   ```
   CREATE ROLE biolabs WITH PASSWORD 'pubtrends';
   CREATE DATABASE pubmed OWNER biolabs; 
   ```
   
3. Make sure that ```crawler/src/main/resources/config.properties``` file contains correct information about the database (url, port, DB name, username and password).
   
   ```
   url = localhost
   port = 5432
   database = pubmed
   username = biolabs
   password = pubtrends
   ```

4. Use the following command to build the project:

   ```
   ./gradlew clean build test
   ```
     
5. Previous command should have produced ```crawler/build/libs/crawler-dev.jar``` file.
   From now on you can use JAR file, no args should be specified when launching the program. 
   First run allows you to download complete up-to-date PubMed database.
   Further runs will allow you to download daily updates.
      
6. Interruption may cause problems with further usage.