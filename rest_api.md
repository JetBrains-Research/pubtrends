# PubTrends REST API (_DRAFT_)

REST API suggests usage of resources. Currently, we have several types of resources that should be accessed through API: topics (defined by search terms, e.g., `brain computer interface`), papers (during paper analysis), data for plots, dataframes (for using in Jupyter Notebook)

**Questionable**:
* Celery's `jobid` can be used as `TOPIC_ID` and `PAPER_ID` currently, however, we can consider reworking it in the future.

## Suggested workflows

1. Web service: load plot data into `bokeh` using `AjaxDataSource` through API.

2. Jupyter notebook: launch search, check status until SUCCESS, then access analyzer. 

## Topic Analysis

### Launch search for certain terms

**Request**

[GET] `/api/topics?source=Pubmed&query=<terms>`

**Response**

`{"topic_id": TOPIC_ID}`

### Check status

**Request**

[GET] `/api/topics/TOPIC_ID/status`

**Response**

`{"status": "SUCCESS" or "PENDING" or "FAILURE"}`

### Get data for a plot
Available only after the search has finished. List of valid PLOT_IDs:

* cocitations_clusters
* component_size_summary 
* component_years_summary_boxplots
* top_cited_papers
* max_gain_papers
* max_relative_gain_papers
* component_sizes
* component_ratio
* papers_stats

**Request**

[GET] `/api/topics/TOPIC_ID/plots/PLOT_ID`

**Response**

Corresponding ColumnDataSource in JSON format.

### Get data for a subtopic plot

**Request**

[GET] `/api/topics/TOPIC_ID/subtopics/SUBTOPIC_ID/`

**Response**

Corresponding ColumnDataSource in JSON format.

### Get data to reproduce KeyPaperAnalyzer in Jupyter Notebook

Available only after the search has finished. 

**Request**

[GET] `/api/topics/TOPIC_ID/analyzer`

**Response**

KeyPaperAnalyzer in JSON format.

## Paper Analysis

### Launch search for a certain paper

**Request**

[GET] `/api/papers?source=Pubmed&key=<key>&value=<value>`

**Response**

`{"paper_id": PAPER_ID}`

### Check status

**Request**

[GET] `/api/papers/PAPER_ID/status`

**Response**

`{"status": "SUCCESS" or "PENDING" or "FAILURE"}`

### Get data for a plot
Available only after the search has finished. List of valid PLOT_IDs:

* citation_dynamics

**Request**

[GET] `/api/topics/TOPIC_ID/plots/PLOT_ID`

**Response**

Corresponding ColumnDataSource in JSON format.

### Get data to reproduce KeyPaperAnalyzer in Jupyter Notebook
Available only after the search has finished. 

**Request**

[GET] `/api/papers/PAPER_ID/analyzer`

**Response**

KeyPaperAnalyzer in JSON format.
