import sys

import pandas as pd
import requests
from Bio import Entrez
from bs4 import BeautifulSoup

BIORXIV_BASE_URL = 'http://biorxiv.org'
Entrez.email = 'nikolay.kapralov@gmail.com'
unique_tags = {}


def title2pmid(title):
    handle = Entrez.esearch(db='pubmed', retmax='10000', retmode='xml', term=title)
    ids = Entrez.read(handle)['IdList']
    if len(ids) == 1:
        return ids[0]
    else:
        return -1


def get_article_data(url):
    response = requests.get(url)
    if not response.ok:
        print(f'Response from server: {response}')
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    id = soup.find('meta', {'name': "citation_id"})['content'].strip().split('v')[0]
    title = soup.find('meta', {'name': "DC.Title"})['content'].strip()
    year = soup.find('meta', {'name': "DC.Date"})['content'].strip()[:4]
    pmid = title2pmid(title)

    return id, title, pmid, year


def get_article_tags(url):
    response = requests.get(url)
    if not response.ok:
        print(f'Response from server: {response}')
        return None

    new_tags_found = False

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup.find_all('meta'):
        try:
            name = tag['name']
            if name not in unique_tags:
                new_tags_found = True
                unique_tags[name] = url
        except KeyError:
            pass

    # Scrape necessary data
    title = soup.find('meta', {'name': "DC.Title"})['content'].strip()

    return new_tags_found


#
# titles = []
# authors = []
# dates = []
# abstracts = []
# links = []
# tags = []
# author_aff = []

def main():
    articles = []
    total_article_count = 0
    articles_with_pmid = 0

    try:
        # TODO: obtain number of pages from HTML source code
        for index in range(2171, 5367):
            print(index, end='')
            if index == 0:
                r = requests.get("http://biorxiv.org/content/early/recent")
            else:
                r = requests.get("http://biorxiv.org/content/early/recent?page=" + str(index))
            if not r.ok:
                print(r)
            soup = BeautifulSoup(r.content, "html.parser")
            # for i in soup.find_all("span", {"class": "highwire-cite-title"})[::2]:
            #     titles.append(i.text.strip())
            # for i in soup.find_all('div', {'class': 'highwire-cite-authors'}):
            #     temp = []
            #     for j, k in zip(i.find_all('span', {'class': 'nlm-given-names'}),
            #                     i.find_all('span', {'class': 'nlm-surname'})):
            #         given = j.text.strip()
            #         surname = k.text.strip()
            #         temp.append(f'{given} {surname}')
            #     authors.append(temp)
            for i in soup.find_all('a', {'class': 'highwire-cite-linked-title'}):
                # links.append(i.get('href').strip())
                print('.', end='')
                url = i.get('href').strip()
                article_data = get_article_data(BIORXIV_BASE_URL + url)
                articles.append(article_data)

                total_article_count += 1
                if article_data[2] != -1:
                    articles_with_pmid += 1

            print(f'({articles_with_pmid} / {total_article_count})')
            sys.stdout.flush()
    except Exception as e:
        print(e)
        articles_df = pd.DataFrame(data=articles, columns=['id', 'title', 'pmid', 'year'])
        articles_df.to_csv('biorxiv-6.csv')

    # print(f'Unique tags found:')
    # print(unique_tags)
    #
    # csv.DictWriter('../../notes/biorxiv-tags-2.tsv', unique_tags, delimiter='\t')
    articles_df = pd.DataFrame(data=articles, columns=['id', 'title', 'pmid', 'year'])
    articles_df.to_csv('biorxiv-5.csv')


main()

# for index, i in enumerate(links):
#     print(index)
#     r = requests.get('http://biorxiv.org' + i)
#     if not r.ok:
#         print(r)
#     soup = BeautifulSoup(r.content)
#     dates.append(soup.find('meta', {'name': "DC.Date"})['content'].strip())
#     abstracts.append(soup.find('meta', {'name': "DC.Description"})['content'].strip())
#     temp = []
#     unique = {}
#     for j in soup.find_all('meta', {'name': "DC.Contributor"}):
#         if j.text.strip() not in unique:
#             unique[j.text.strip()] = ''
#             temp.append(j.text.strip())
#     author_aff.append(temp)
#     temp = []
#     # for j in soup.find_all('span', {'class': 'highwire-article-collection-term'}):
#     #     temp.append(j.text.strip())
#     tags.append(temp)
#
# if len(titles) == len(authors) == len(dates) == len(abstracts) == len(links) == len(tags) == len(author_aff):
#     f = open('biorxiv.txt', 'w')
#     for title, author, date, abstract, link, tag, author_af in zip(titles, authors, dates, abstracts, links, tags,
#                                                                    author_aff):
#         f.write(str([title, author, date, abstract, link, tag, author_af]))
#         f.write('\n')
#     f.close()
# else:
#     print('error')
