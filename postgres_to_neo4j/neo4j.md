Neo4j integration
=================
This document contains brief instruction on configuring [Neo4j](https://neo4j.com/product/?ref=home-banner) database for Pubtrends project.

Configure Neo4j
---------------

* Export data from PostgreSQL to csv format

    ```
    psql postgres://user:password@host:5432/pubtrends -f export_to_csv.psql;

    ```
  
Data should be post processed:
    
    ```
    cat pmpublications.tsv | sed "s#\"\"#'#g" | sed 's#"##g' > pmpublications_quotes.tsv
    cat sspublications.tsv | sed "s#\"\"#'#g" | sed 's#"##g' > sspublications_quotes.tsv
    ```  

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
    
* Launch Neo4j database as a service
    
    ```
    docker run --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/data --volume=$HOME/neo4j/logs:/logs neo4j:3.5
    ```

* Open `localhost:7474` in web browser, default login/password are `neo4j/neo4j`.


Playground with Pubmed
----------------------

* Create index on `pmid`

    ```
    CREATE INDEX ON :PMPublication(pmid);
    ```

* Lookup publications by "DNA methylation clock" Entrez query (partly) 
    
    ```
    WITH ['16999817', '16717091', '16683245', '16582617', '16314580', '15975143', '15941485', '15860628', '15790588', 
          '15779908', '14577056', '11820819', '11032969', '1943146', '1722018', '2777259', '2857475'] AS pmids 
    MATCH (p:PMPublication) 
    WHERE p.pmid IN pmids 
    RETURN p
    LIMIT 20;
    ```

* Build full-text-search index for title and abstract records. **NOTE**: this is time and resources - consuming operation

    ```
    CALL db.index.fulltext.createNodeIndex("pmTitlesAndAbstracts",["PMPublication"],["title", "abstract"]);
    ```

* Inspect search results

    ```
    CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", "methylation") YIELD node, score 
    RETURN node.title, node.abstract, score LIMIT 20;
    ```

* Compute amount of results
    ```
    CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", "cancer") YIELD node RETURN count(*);
    ```

* Build search with different search strategies supported by Pubtrends

Most cited:

    ```
    CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", "'human' AND 'aging'") YIELD node AS n
    MATCH ()-[r:PMReferenced]->(p:PMPublication) 
    WHERE p.pmid = n.pmid 
    WITH p, COUNT(r) AS cnt 
    RETURN p 
    ORDER BY cnt DESC 
    LIMIT 100;
    ```

Most recent:

    ```
    CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", "'human' AND 'aging'") YIELD node AS n
    RETURN n 
    ORDER BY n.date DESC 
    LIMIT 100;
    ```
    
Most relevant:

    ```
    CALL db.index.fulltext.queryNodes("pmTitlesAndAbstracts", "'human' AND 'aging'") YIELD node AS n, score AS s
    RETURN n 
    ORDER BY s DESC 
    LIMIT 100;
    ```