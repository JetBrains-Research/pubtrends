Pubtrends changelog
===================

Here you can see the full list of changes between each release.

Version 1.3
------------
Released on XXX XX, 2025

- Reworked text embeddings usage for global similarity network construction
- Supported Sentence Transformer models for whole texts embeddings
- Implemented Questions for paper feature using improved embeddings
- Preview of Semantic search with Sentence Transformer for embeddings, Postgresql with vector extension for embeddings storage and Faiss for fast embeddings lookup 

Version 1.2
------------
Released on April 3, 2025

- Improved lookup for frequent words, e.g. cell, human, cancer, etc.
- Introduced analysis for group of papers

Version 1.1
------------
Released on March 14, 2025

- Use weighted sum of graph and text embeddings instead of concatenation
- Improved graph visualization - limit number of show connected papers
- Updated databases
- Updated libraries
- Docker base image updated to Ubuntu 24.04 LTS


Version 1.0
------------
Released on June 29, 2023

- All search results are saved permanently
- Use different sparse graphs for embeddings and visualization
- Use kNN induced graph to create sparse graph
- Tweaked analysis parameters for better clusters extraction
- Updated libraries


Version 0.23
------------
Released on May 10, 2023

- Cleanup tasks cache - removed lru_ttl_cache_with_callback
- Don't use fasttext model, it requires lots of memory without significant improvements
- Docker base image updated to Ubuntu 22.04 LTS

Version 0.22
------------
Released on Aug 23, 2022

- Updated Semantic Scholar database to version 2022-05-01
- Reworked Semantic Scholar database structure and indexes for faster loading and updates
- Added size selector in network view - size encodes number of citations or centrality of the paper
- Don't show feedback message form for predefined examples
- Small bugfixes in numbers extracting functionality
- Better tests on Kotlin-based database uploading and Python-based analysis
- Updated libraries versions to support MacBook aarch64 architecture

Version 0.21
------------
Released on Jan 25, 2022

- Updated PubMed database to the version 2022
- Updated config to use nginx
- Improved single paper analysis - pay more attention to paper own keywords/citations while expanding by references
- Show papers graph for single-paper analysis
- Fixed issues with missing references for fresh papers

Version 0.20
------------
Released on Nov 18, 2021

- Use pretrained fasttext model by Facebook for words embeddings
- Reworked text preprocessing - don't ignore non-ascii characters
- Optimize memory consumption for workers

Version 0.19
------------
Released on Oct 10, 2021

- Use combined text and graph embeddings for papers analysis
- Get rid of citations graph visualization
- Disabled zoom out functionality, reworked execution
- Fixed review generation functionality
- Reduced default font size
- Various other bugfixes

Version 0.18
------------
Released on Oct 5, 2021

- Support Pubmed search syntax including AND, OR, etc.
- Show analysis results as downloadable html files
- Reworked single paper analysis with single background task
- Small fixes

Version 0.17
------------
Released on Aug 10, 2021

- Improved topics descriptions based on cosine similarities
- Use similarity and clustering parameters according to the nature reviews benchmark,
   see https://dl.acm.org/doi/10.1145/3459930.3469501
- Papers filtering in graph representation based on specific fields
- Show topics centers and tags in interactive viewer
- Small updates and bugfixes


Version 0.16
------------
Released on May 24, 2021

- Use graph embeddings for detecting similar papers
- Improved papers graph visualization in tSNE coordinates of embeddings
- Identify groups of similar authors
- Small updates and bugfixes


Version 0.15
------------
Released on April 9, 2021

- Removed Neo4j DB backend

Version 0.14
------------
Released on April 7, 2021

- Reworked results page
- Show joint topics and keyword diagram
- Bugfixes

Version 0.13
------------
Released on March 28, 2021

 - Analyze list of papers
 - Improved papers search - combine pharases with terms search
 - Bugfixes


Version 0.12
------------
Released on March 1, 2021

 - Improved graph visualization - highlight nodes according to filter string on the fly
 - Improved topics description - avoid similar terms in topics descriptions
 - Modified similarity between papers - use log for bibcoupling and co-citations to make all features the same magnitude


Version 0.11
------------
Released on February 12, 2021

 - Model from the paper [Automatic generation of reviews of scientific papers](https://arxiv.org/abs/2010.04147) is deployed. \
 It collects a set of sentences from top cited papers abstracts with the highest probability to be included in a real review
 - Feedback frontend code cleanup
 - Export results
 - Configure additional features from config file

Version 0.10
-------------

Released on November 22, 2020

 - Redesign of main page
 - Add dedicated on-site webpage with example search description
 - HTTPS supported
 - Add Google Analytics code during building
 - robots.txt and sitemap.xml


Version 0.9
-------------

Released on October 22, 2020

 - Support for feedback collecting
 - Changed expand logic: keep total citation count the same magnitude
 - Small Bugfixes


Version 0.8
-------------

Released on September 24, 2020

 - Extract numbers from texts of abstracts
 - Small Bugfixes


Version 0.7
-------------

Released on September 1, 2020

 - Reworked DB backends (default is postgres)
 - Optimized most cited search for postgres
 - Support MESH terms and keywords for Pubmed
 - Expand queries when small number of papers can be found by direct search
 - Show keywords and MESH terms in paper info
 - Bugfixes and improvements


Version 0.6
-------------

Released on August 8, 2020

 - Reworked topics extraction based on the smallest communities of Louvain algorithm
 - Removed limitation of 20 topics
 - Show highly similar papers between topics in structure graph
 - Bugfixes


Version 0.5
-------------

Released on May 3, 2020

 - Export to csv functionality for papers list
 - Reworked logging and configuration files lookup
 - Initial user management
 - Admin interface with visits, searches statistics and requests word cloud


Version 0.4
-------------

Released on April 27, 2020

 - Search by DOI supported
 - Show DOI information for single and multiple papers
 - Fixes in papers expand


Version 0.3
-------------

Released on Mar 20, 2020

 - Introduced cytoscape based views for citations / structure graphs
 - Hybrid citations based: citations based, co-citations and bibliographic coupling.
 - Use TF-IDF for topics description


Version 0.2
-------------

Released on January 9, 2020

 - Switched to Neo4j instead of PostgreSQL


Version 0.1
-------------

Released on July 25, 2019

 - Docker-compose deployment
 - Semantic Scholar and Pubmed supported
 - Initial release
