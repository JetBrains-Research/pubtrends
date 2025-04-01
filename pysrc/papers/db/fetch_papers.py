# Small example how to load papers information from the database
import pandas as pd
import psycopg2

# TODO fill
host='localhost'
port='port'
dbname='dn'
user='user'
password='password'
connection_string = f"""
                    host={host} \
                    port={port} \
                    dbname={dbname} \
                    user={user} \
                    password={password}
                """.strip()
postgres_connection = psycopg2.connect(connection_string)

query="""
SELECT P.pmid as id, title, abstract, year, doi, aux
FROM PMPublications P
WHERE P.pmid IN (VALUES 
(30660649),(29895827),(29121253)
);
"""

with postgres_connection.cursor() as cursor:
    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall(), columns=['id', 'title', 'abstract', 'year', 'doi', 'aux'], dtype=object)

def extract_authors(authors_list):
    if not authors_list:
        return ''
    return ', '.join(filter(None, map(lambda authors: authors['name'], authors_list)))


df['aux'] = df['aux'].apply(lambda aux: json.loads(aux) if type(aux) is str else aux)
df['authors'] = df['aux'].apply(lambda aux: extract_authors(aux['authors']))
df['journal'] = df['aux'].apply(lambda aux: aux['journal']['name'])
df.drop(['aux'], axis=1, inplace=True)

df.to_csv('~/papers.csv')