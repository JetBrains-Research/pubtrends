Playground with Pubmed
----------------------

* Download Pubmed papers using instructions from [README.md](README.md)

* Open neo4j `localhost:7474` web browser.

* Lookup publications by "DNA methylation clock" Entrez query (partly) 
    
    ```
    WITH ['16999817', '16717091', '16683245', '16582617', '16314580', '15975143', '15941485', '15860628', '15790588', 
          '15779908', '14577056', '11820819', '11032969', '1943146', '1722018', '2777259', '2857475'] AS pmids 
    MATCH (p:PMPublication) 
    WHERE p.pmid IN pmids 
    RETURN p
    LIMIT 20;
    ```

* Inspect full text search results

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
