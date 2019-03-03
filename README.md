# PubTrends

A tool for analysis of trends & pivotal points in the scientific literature.

Project workflow: https://drive.google.com/open?id=17oQAuMJ0vmDgzucGxJT_xgQBCkLEio3HIeG_sfgMtcc

### Prerequisites

* JDK 8
* PostgreSQL (tested with 10.5)

### Getting started

1. Clone the repository.

2. Run `psql` to create a user and a database in PostgreSQL:

   ```
   CREATE ROLE biolabs WITH PASSWORD 'pubtrends';
   ALTER ROLE "biolabs" WITH LOGIN;
   CREATE DATABASE pubmed OWNER biolabs; 
   ```
   
3. Make sure that `crawler/src/main/resources/config.properties` file contains correct information about the database (url, port, DB name, username and password).
   
   ```
   url = localhost
   port = 5432
   database = pubmed
   username = biolabs
   password = pubtrends
   ```

4. Other configuration parameters in `crawler/src/main/resources/config.properties`:

   * `lastId` - in case of interruption use this parameter to restart the download from article pack `pubmed18n{lastId+1}.xml` (should be done automatically later) 
   * `parserLimit` - maximum number of articles per XML to be parsed (useful for development)
   * `resetDatabase` - clear current contents of the database (useful for development)
   * `gatherStats` - count number of XML tag occurences (useful for development)

5. Use the following command to build the project:

   ```
   ./gradlew clean build test
   ```
     
6. Previous command should have produced `crawler/build/libs/crawler-dev.jar` file.
   From now on you can use JAR file, no args should be specified when launching the program. 
   First run allows you to download complete up-to-date PubMed database.
   Further runs will allow you to download daily updates.
      
7. In case of interruption use `lastId` parameter as described above. Interruption can be caused by internet connection problems.
