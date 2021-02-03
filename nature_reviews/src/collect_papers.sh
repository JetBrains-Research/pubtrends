# This script is used to collect publications from 5 Nature Review reviews,
# which have references information in PubMed database.
# Journals used:
# * https://www.nature.com/nrc/
# * https://www.nature.com/nrg/
# * https://www.nature.com/nri/
# * https://www.nature.com/nrm/
# * https://www.nature.com/nrn/

# Fetch different years and journals from Postgresql into .txt files
# Please ensure that you've configured env variable with postgresql password
for YEAR in 2016 2017 2018 2019 2020; do
  for JOURNAL in "Cancer" "Immunology" "Molecular cell biology" "Neuroscience"; do
    echo "$YEAR $JOURNAL";
    CMD="WITH X AS (SELECT pmid_out, count(1) FROM PMCitations C GROUP BY pmid_out) \
      SELECT P.pmid, P.title \
      FROM PMPublications P \
      INNER JOIN X ON P.pmid = X.pmid_out \
      WHERE P.year = $YEAR \
        AND X.count >= 10 AND \
        P.type='Review' AND \
        P.aux->'journal'->>'name'='Nature reviews. $JOURNAL' \
      LIMIT 4;";
    echo "$CMD";
    psql -h franklin.labs.intellij.net -p 5432 -U biolabs -d pubtrends -q -o "$YEAR Nature reviews. $JOURNAL.txt" -c "$CMD";
  done;
done;

# Collect separate journal files .txt files into a single .txt file
for YEAR in 2016 2017 2018 2019 2020; do
  for JOURNAL in "Cancer" "Immunology" "Molecular cell biology" "Neuroscience"; do
    echo "$YEAR $JOURNAL";
    cat "$YEAR Nature reviews. $JOURNAL.txt" | grep -v pmid | grep -v rows |\
      WHILE read -r $LINE; do
        echo "$YEAR | Nature reviews. $JOURNAL | $LINE" >> result.txt;
      done;
  done;
done;

# Transform single .txt file into a .CSV file
rm result.csv;
for YEAR in 2016 2017 2018 2019 2020; do
  for JOURNAL in "Cancer" "Immunology" "Molecular cell biology" "Neuroscience"; do
    echo "$YEAR $JOURNAL";
    cat "$YEAR Nature reviews. $JOURNAL.txt" | grep -v pmid | grep -v rows | grep -v -e "---" |\
      xargs -I {} echo "$YEAR | Nature reviews. $JOURNAL | {}" | sed 's/ | /,/g' >> result.csv;
  done;
done;
