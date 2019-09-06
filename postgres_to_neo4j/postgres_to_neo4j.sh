# export from postgres to csv
psql postgres://biolabs:password@franklin.labs.intellij.net:5432/pubtrends -f export_twitter_to_csv.psql;

#run neo4j in Docker
docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data --volume=$HOME/neo4j/logs:/logs neo4j:3.5

#import into neo4j
docker run --volume=/home/user/work/pubtrends:/pubtrends  \
  --volume=$HOME/neo4j/data:/data \
  --volume=$HOME/neo4j/logs:/logs \
   -it neo4j:3.5 /bin/bash

# inside container
neo4j-admin import -ignore-missing-nodes=true --mode csv --multiline-fields \
  --nodes="/pubtrends/pmpublications_header.csv,/pubtrends/pmpublications.csv" \
  --relationships:References="/pubtrends/pmcitations_header.csv,/pubtrends/pmcitations.csv"
