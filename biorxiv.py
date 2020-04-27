import time
import requests
from bs4 import BeautifulSoup
import unicodedata

titles = []
authors = []
dates = []
abstracts = []
links = []
tags = []
author_aff = []

for index in range(10):
    print(index)
    if index == 0:
        r = requests.get("http://biorxiv.org/content/early/recent")
    else:
        r = requests.get("http://biorxiv.org/content/early/recent?page=" + str(index))
    if not r.ok:
        print(r)
    soup = BeautifulSoup(r.content)
    for i in soup.find_all("span", {"class": "highwire-cite-title"})[::2]:
        titles.append(i.text.strip())
    for i in soup.find_all('div', {'class': 'highwire-cite-authors'}):
        temp = []
        for j, k in zip(i.find_all('span', {'class': 'nlm-given-names'}), i.find_all('span', {'class': 'nlm-surname'})):
            given = unicodedata.normalize('NFKD', j.text.strip()).encode('ascii', 'ignore')
            sur = unicodedata.normalize('NFKD', k.text.strip()).encode('ascii', 'ignore')
            temp.append(f'{given} {sur}')
        authors.append(temp)
    for i in soup.find_all('a', {'class': 'highwire-cite-linked-title'}):
        links.append(i.get('href').strip())

for index, i in enumerate(links):
    print(index)
    r = requests.get('http://biorxiv.org' + i)
    if not r.ok:
        print(r)
    soup = BeautifulSoup(r.content)
    dates.append(soup.find('li', {'class': "published"}).text.strip('Posted').strip())
    abstracts.append(soup.find('p', {'id': "p-2"}).text.strip())
    temp = []
    unique = {}
    for j in soup.find_all('span', {'class': 'nlm-aff'}):
        if j.text.strip() not in unique:
            unique[j.text.strip()] = ''
            temp.append(j.text.strip())
    author_aff.append(temp)
    temp = []
    for j in soup.find_all('span', {'class': 'highwire-article-collection-term'}):
        temp.append(j.text.strip())
    tags.append(temp)

if len(titles) == len(authors) == len(dates) == len(abstracts) == len(links) == len(tags) == len(author_aff):
    f = open('biorxiv.txt', 'w')
    for title, author, date, abstract, link, tag, author_af in zip(titles, authors, dates, abstracts, links, tags,
                                                                   author_aff):
        f.write(str([title, author, date, abstract, link, tag, author_af]))
        f.write('\n')
    f.close()
else:
    print('error')
