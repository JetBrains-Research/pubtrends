# Database Configuration

## PostgreSQL Setup

### Initialize Main Database

1. Launch Docker image:
```bash
docker run --rm --name pubtrends-postgres \
    -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
    -v ~/postgres/:/var/lib/postgresql/data \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -p 5432:5432 \
    -d postgres:17
```

2. Create a database (once a database is created use `-d pubtrends` argument):
```bash
psql -h localhost -p 5432 -U biolabs
ALTER ROLE biolabs WITH LOGIN;
CREATE DATABASE pubtrends OWNER biolabs;
```

3. Configure memory params in `~/postgres/pgdata/postgresql.conf`:
```
# Memory settings
effective_cache_size = 8GB  # ~ 50 to 75% (can be set precisely by referring to "top" free+cached)
shared_buffers = 2GB        # ~ 1/4 – 1/3 total system RAM
work_mem = 1GB              # For sorting, ordering etc
max_connections = 4         # Total mem is work_mem * connections
maintenance_work_mem = 1GB  # Memory for indexes, etc

# Write performance
checkpoint_timeout = 10min
checkpoint_completion_target = 0.8
synchronous_commit = off
```

You can check current settings by command `SHOW ALL;` in psql console.

### PostgreSQL with Vector Extension

For semantic search and embeddings, launch pgvector:

```bash
docker run --rm --name pgvector -p 5430:5432 \
    -m 32G \
    -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
    -e POSTGRES_DB=pubtrends \
    -v ~/pgvector/:/var/lib/postgresql/data \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -d pgvector/pgvector:pg17
```

## Papers Downloading and Processing

PostgreSQL should be configured and launched before proceeding.

### Pubmed

Launch crawler to download and keep up to date a Pubmed database:

```bash
java -cp build/libs/pubtrends-dev.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase
```

Command line options supported:
* `resetDatabase` - clear current contents of the database (for development)
* `fillDatabase` - option to fill a database with Pubmed data. Can be interrupted at any moment.
* `lastId` - force downloading from given id from articles pack `pubmed20n{lastId+1}.xml`.

Updates - add the following line to crontab:

```bash
crontab -e
0 22 * * * java -cp pubtrends-<version>.jar org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase | \
tee -a crontab_update.log
```

### Semantic Scholar

Download Sample from [Semantic Scholar](https://www.semanticscholar.org/) or full archive. See Open Corpus.
The latest release can be found at: https://api.semanticscholar.org/api-docs/datasets#tag/Release-Data

```bash
curl https://api.semanticscholar.org/datasets/v1/release/
```

#### Linux & Mac OS

```bash
# Fail on errors
set -euox pipefail

DATE="2022-05-01"
PUBTRENDS_JAR=

wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/manifest.txt
echo "" > complete.txt
N=$(cat manifest.txt | grep corpus | wc -l)
cat manifest.txt | grep corpus | while read -r file; do
   if [[ -z $(grep "$file" complete.txt) ]]; then
      echo "Processing $file / $N"
      wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/$file;
      java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase $(pwd)/$file
      rm $file;
      echo "$file" >> complete.txt
   fi;
done
java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --index --finish
```

#### Windows 10 PowerShell

```powershell
$DATE = "2023-03-14"
$PUBTRENDS_JAR =
curl.exe -o .\manifest.txt https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/manifest.txt
echo "" > .\complete.txt
foreach ($file in Get-Content .\manifest.txt) {
    $sel = Select-String -Path .\complete.txt -Pattern $file
    if ($sel -eq $null) {
       echo "Processing $file"
       curl.exe -o .\$file https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/$DATE/$file
       java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --fillDatabase .\$file
       del ./$file
       echo $file >> .\complete.txt
    }
}
java -cp $PUBTRENDS_JAR org.jetbrains.bio.pubtrends.ss.SemanticScholarLoader --index --finish
```

## Updating Embeddings

Please ensure that embeddings Postgres DB with vector extension is up and running (see PostgreSQL with Vector Extension section).

Then you'll be able to update embeddings with a commandline below. It will compute embeddings and store them into the vector DB, and update the Faiss index for fast search.

```bash
docker build -f pysrc/preprocess/embeddings/Dockerfile -t update_embeddings --platform linux/amd64 .
docker run  -v ~/.pubtrends:/config:ro \
   -v ~/.pubtrends/logs:/logs \
   -v ~/.pubtrends/sentence-transformers:/sentence-transformers \
   -v ~/.pubtrends/nltk_data:/home/user/nltk_data \
   -v ~/.pubtrends/faiss:/faiss \
   -it update_embeddings /bin/bash

uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --no-cache sentence-transformers faiss-cpu
export PYTHONPATH=$PYTHONPATH:$(pwd)
/bin/bash scripts/nlp.sh
python pysrc/preprocess/update_embeddings.py
```

## Deployment

### PostgreSQL for Production

1. Start Postgres server:

```bash
docker run --rm --name pubtrends-postgres -p 5432:5432 \
    -m 32G \
    -e POSTGRES_USER=biolabs -e POSTGRES_PASSWORD=mysecretpassword \
    -e POSTGRES_DB=pubtrends \
    -v ~/postgres/:/var/lib/postgresql/data \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -d postgres:17
```

**NOTE**: stop Postgres docker image with timeout `--time=300` to avoid DB recovery.

**NOTE2**: for speed reasons we use materialize views, which are updated upon a successful database update. In case of an emergency stop, the view should be refreshed manually to ensure sort by citations works correctly:

```bash
psql -h localhost -p 5432 -U biolabs -d pubtrends
refresh materialized view matview_pmcitations;
```
