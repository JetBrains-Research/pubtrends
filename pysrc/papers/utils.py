import binascii
import logging
import re
from string import Template

import pandas as pd
from matplotlib import colors

PUBMED_ARTICLE_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='
SEMANTIC_SCHOLAR_BASE_URL = 'https://www.semanticscholar.org/paper/'

# IMPORTANT
# KeyPaperAnalyzer.launch() performs "zoom - 1" expand operations if id_list is given.
# This allows to perform zoom out in case of regular zoom out action and double in case of single paper analysis.
ZOOM_IN = 0
ZOOM_OUT = 1
PAPER_ANALYSIS = 2

ZOOM_IN_TITLE = 'detailed'
ZOOM_OUT_TITLE = 'expanded'
PAPER_ANALYSIS_TITLE = 'paper-analysis'

SORT_MOST_CITED = 'Most Cited'
SORT_MOST_RECENT = 'Most Recent'

MAX_TITLE_LENGTH = 200

SEED = 19700101

log = logging.getLogger(__name__)


def zoom_name(zoom):
    if int(zoom) == ZOOM_IN:
        return ZOOM_IN_TITLE
    elif int(zoom) == ZOOM_OUT:
        return ZOOM_OUT_TITLE
    elif int(zoom) == PAPER_ANALYSIS:
        return PAPER_ANALYSIS_TITLE
    raise ValueError(f'Illegal zoom key value: {zoom}')


def cut_authors_list(authors, limit=10):
    # handle empty string and float('nan') cases
    if not authors or pd.isnull(authors):
        return "No authors listed"

    before_separator = limit - 1
    separator = ',...,'
    author_list = authors.split(', ')
    if len(author_list) > limit:
        return ', '.join(author_list[:before_separator]) + separator + author_list[-1]
    return authors


def extract_authors(authors_list):
    if not authors_list:
        return ''

    return ', '.join(filter(None, map(lambda authors: authors['name'], authors_list)))


def crc32(hex_string):
    n = binascii.crc32(bytes.fromhex(hex_string))
    return to_32_bit_int(n)


def to_32_bit_int(n):
    if n >= (1 << 31):
        return -(1 << 32) + n
    return n


def trim(string, max_length):
    return f'{string[:max_length]}...' if len(string) > max_length else string


def preprocess_doi(line):
    """
    Removes doi.org prefix if full URL was pasted, then strips unnecessary slashes
    """
    (_, _, doi) = line.rpartition('doi.org')
    return doi.strip('/')


def preprocess_search_title(line):
    """
    Title processing similar to PubmedXMLParser - special characters removal
    """
    return line.strip('.[]')


def rgb2hex(color):
    if isinstance(color, str):
        match = re.match('rgb\\((\\d+), (\\d+), (\\d+)\\)', color)
        if match:
            r, g, b = match.group(1), match.group(2), match.group(3)
        else:
            r, g, b, _ = colors.to_rgba(color)
            r, g, b = r * 255, g * 255, b * 255
    else:
        r, g, b = color
    return "#{0:02x}{1:02x}{2:02x}".format(int(r), int(g), int(b))


def hex2rgb(color):
    return [int(color[pos:pos + 2], 16) for pos in range(1, 7, 2)]

