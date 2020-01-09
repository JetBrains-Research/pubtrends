Neo4j integration
=================
This document contains brief instruction on configuring [Neo4j](https://neo4j.com/product/?ref=home-banner) database for Pubtrends project.

Export from Postgresql
---------------

* Export data from PostgreSQL to csv format

    ```
    psql postgres://user:password@host:5432/pubtrends -f export_to_csv.psql;

    ```
  
Import into Postgresql
---------------
  
* Connect into Neo4j Docker container interactively 

    ```
    docker run --volume=/home/user/work/pubtrends:/pubtrends  \
        --volume=$HOME/neo4j/data:/data --volume=$HOME/neo4j/logs:/logs \
        -it neo4j:3.5 /bin/bash
    ```
    
* Launch batch import procedure (within Docker container)

Pubmed:

    ```
    neo4j-admin import -ignore-missing-nodes=true --mode csv --delimiter='\t' --multiline-fields \
        --nodes:PMPublication="/pubtrends/pmpublications_header.tsv,/pubtrends/pmpublications.tsv" \
        --relationships:PMReferenced="/pubtrends/pmcitations_header.tsv,/pubtrends/pmcitations.tsv"
    ```
  
Semantic Scholar:

    ```
    neo4j-admin import -ignore-missing-nodes=true --mode csv --delimiter='\t' --multiline-fields \
        --nodes:SSPublication="/pubtrends/sspublications_header.tsv,/pubtrends/sspublications.tsv" \
        --relationships:SSReferenced="/pubtrends/sscitations_header.tsv,/pubtrends/sscitations.tsv"
    ```
Load into neo4j:

    ```
    USING PERIODIC COMMIT
    LOAD CSV FROM "file:///sspublications_quotes.tsv" AS line FIELDTERMINATOR '\t'
    CREATE (:SSPublication {ssid: line[0], pmid: line[1], title: line[2], abstract: line[3], 
                            year:toInteger(line[4]), source: line[5], keywords: line[6], aux: line[7]})
    ```
    
    ```
    USING PERIODIC COMMIT
    LOAD CSV FROM "file:///sscitations.tsv" AS line AS line FIELDTERMINATOR '\t'
    MATCH (out:SSPublication),(in:SSPublication)
    WHERE out.ssid = line[0] AND in.ssid = line[1]
    CREATE (out)-[r:SSReferenced]->(in)
    ```
