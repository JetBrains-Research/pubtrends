[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Build Status](http://teamcity.jetbrains.com/app/rest/builds/buildType:(id:BioLabs_Pubtrends_Deployment)/statusIcon.svg)](http://teamcity.jetbrains.com/viewType.html?buildTypeId=BioLabs_Pubtrends_Deployment&guest=1)
[![DOI](https://zenodo.org/badge/151591143.svg)](https://doi.org/10.5281/zenodo.15131474)

PubTrends
=========

PubTrends is an interactive scientific literature exploration tool that helps researchers analyze topics, visualize
research trends, and discover related works.

Available online at: https://pubtrends.info/

# Overview

With PubTrends, you can:

* Gain a concise overview of your research area.
* Explore popular trends and impactful publications.
* Discover new and promising research directions.

See an example of the analysis at: https://pubtrends.info/about.html

# Datasets:

* [Pubmed](https://pubmed.ncbi.nlm.nih.gov) ~40 mln papers and 450 mln citations
* [Semantic Scholar](https://www.semanticscholar.org) 170 mln papers and 600 mln citations

![Scheme](pysrc/app/static/about/about_pubtrends_scheme.png?raw=true "Title")

# Technical Architecture

PubTrends is a Python / Kotlin + JavaScript web service with a PostgreSQL backend.
It uses:
* Languages: [Python](https://www.python.org/) + [Kotlin](https://kotlinlang.org/) + [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
* Backend: [Nginx](https://nginx.org/en/index.html) + [Flask](https://flask.palletsprojects.com/en/stable/) + [Gunicorn](https://gunicorn.org/)
* Task Queue: [Celery](https://docs.celeryq.dev/en/stable/index.html) + [Redis](https://redis.io/)
* DataBase: [Postgres](https://www.postgresql.org/) + [Postgres pgvector](https://github.com/pgvector/pgvector) + [Psycopg2](https://www.psycopg.org/docs/) + [Kotlin ORM](https://www.jetbrains.com/exposed/)
* Data Analysis: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/stable/index.html#)
* Semantic Search: [Sentence-Tranformers](https://www.sbert.net/) + [Faiss](https://github.com/facebookresearch/faiss)
* NLP: [NLTK](https://www.nltk.org/), [SpaCy](https://spacy.io/), [GenSim](https://radimrehurek.com/gensim/models/word2vec.html), [Fasttext](https://fasttext.cc/)
* Visualization: [Bokeh](https://bokeh.org/), [Holoviews](https://holoviews.org/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)
* Frontend: [Bootstrap](https://getbootstrap.com/), [jQuery](https://jquery.com/), [Cytoscape.js](https://js.cytoscape.org/)
* Deployment: [Docker Compose](https://docs.docker.com/compose/)
* Testing: [PyTest](https://docs.pytest.org/en/stable/) + [Flake8](https://flake8.pycqa.org/en/latest/) + [JUnit](https://junit.org/) + [TeamCity](https://www.jetbrains.com/teamcity/)

See [pyproject.toml](pyproject.toml) for the full list of libraries used in the project.

# Getting Started

For detailed information, see:
* [DATABASE.md](docs/DATABASE.md) - Database setup and data loading
* [DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development environment, testing, and deployment
* [CHANGES.md](docs/CHANGES.md) - Version history and changelog

## Quick Start

1. Copy and modify `config.properties` to `~/.pubtrends/config.properties` with your database credentials.

2. Set up the database - see [DATABASE.md](docs/DATABASE.md) for detailed instructions.

3. Set up development environment - see [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed instructions.

# Authors

See [AUTHORS.md](docs/AUTHORS.md) for a list of authors and contributors.

# Materials

* *Shpynov, O. and Kapralov, N., 2021, August. PubTrends: a scientific literature explorer. In Proceedings of the
  12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (pp.
  1-1).* https://doi.org/10.1145/3459930.3469501

* [Icons by Feather](https://feathericons.com/)

# Contributing

Here’s how you can help:

* ⭐ Star this repo, help others to discover it
* 🐛 Found a bug? Open an issue
* 💡 Have an idea? Feel free to submit a feature request or a PR
* 👍 Upvote issues you care about, help us prioritize
