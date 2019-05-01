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

4. Command line options supported:

   * `lastId` - in case of interruption use this parameter to restart the download from article pack `pubmed18n{lastId+1}.xml` 
   * `parserLimit` - maximum number of articles per XML to be parsed (useful for development)
   * `resetDatabase` - clear current contents of the database (useful for development)
   
Interruption can be caused by internet connection problems.
In case of interruption `lastId` parameter should be saved to `~/.pubtrends/crawler`, but can be configured explicitly. 

5. Use the following command to test the project:

   ```
   ./gradlew clean test shadowJar
   ```
     
6. Previous command should have produced `crawler/build/libs/crawler-dev.jar` file.
   From now on you can use JAR file, no args should be specified when launching the program. 
   First run allows you to download complete up-to-date PubMed database.
   Further runs will allow you to download daily updates.
