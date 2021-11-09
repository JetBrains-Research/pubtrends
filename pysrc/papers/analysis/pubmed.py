from Bio import Entrez

from pysrc.papers.config import PubtrendsConfig

# Configure email to get notifications on too heavy API usage
Entrez.email = PubtrendsConfig(test=False).entrez_email


def pubmed_search(query, sort,  limit):
    if sort == 'Most Relevant':
        handle = Entrez.esearch(db='pubmed', retmax=str(limit), retmode='xml', term=query, sort='relevance')
    elif sort == 'Most Recent':
        handle = Entrez.esearch(db='pubmed', retmax=str(limit), retmode='xml', term=query)
    else:
        raise Exception(f'Unknown Pubmed sort option: {sort}')
    return Entrez.read(handle)['IdList']

