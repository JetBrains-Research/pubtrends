## Notes on arXiv Preprint Crawler

#### Rate limits

Current [rate limits](https://arxiv.org/help/api/tou) of arXiv (as of 2020/11/01):
 
 * OAI-PMH, arXiv API - 1 request / 4 s, single connection  
 * Services via the arXiv API Gateway (what is this?) - 4 requests / s, 4 connections
 
#### OAI-PMH vs arXiv API
 
OAI-PMH provides option to gather all articles from certain date, which is good for harvesting.
However, there are no links to PDFs of retrieved papers, which are crucial for reference parsing.
In order to get these links, one should use arXiv API, and it is sad, because presence of PDF link is the only major 
difference in outputs of OAI-PMH and arXiv API.

**Current implementation:**

 * Use OAI-PMH to get all articles that were updated since YYYY-MM-DD. Update may involve only metadata, not necessarily
 a new PDF version. 
 * Filter out papers with updated PDF files.
 * Use arXiv API in order to get links to PDFs.

**Could we use arXiv API for harvesting?**

Probably yes, but with a reversed and more complicated logic - instead of making a query for all articles since YYYY-MM-DD, we would need to query all articles in a certain category.
 
 * pro: avoid using 2 APIs
 * pro: target new versions, not metadata updates
 * con: need to make separate query for each arXiv set
 * con: probably involves high computational load for arXiv servers