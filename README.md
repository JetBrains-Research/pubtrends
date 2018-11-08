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
   
3. In the project folder run the following command to obtain ```crawler-dev.jar``` file in ```build/libs``` folder:

   ```
   ./gradlew crawler:shadowJar
   ``` 
   
4. From now on you can use JAR archive, no args should be specified when launching the program. 
   First run allows you to download complete up-to-date PubMed database.
   Further runs will allow you to download daily updates.
   
4. Current implementation has several in-code parameters (to be refactored later):

   * ```PubmedCrawler.kt:12``` - set username and password to access database, use ```reset``` to clean database while testing:
   
   ```
   private val dbHandler = DatabaseHandler("biolabs", "pubtrends", reset = false)
   ```
   
   * ```PubmedFTPHandler.kt:10``` - if ```limit``` is equal to zero, then everything will be parsed, otherwise only first ```limit``` articles: 
   
   ```
   class PubmedXMLHandler(private val limit : Int = 10) : DefaultHandler() {
   ```
   
   * Interruption may cause problems with further usage.