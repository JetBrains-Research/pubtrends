import logging
import re

import binascii
import pandas as pd
from bokeh.colors import RGB
from bokeh.transform import factor_cmap
import matplotlib as plt
import numpy as np
from matplotlib import colors

PUBMED_ARTICLE_BASE_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/?term='
SEMANTIC_SCHOLAR_BASE_URL = 'https://www.semanticscholar.org/paper/'

IDS_ANALYSIS_TYPE = 'detailed'
PAPER_ANALYSIS_TYPE = 'paper'

SORT_MOST_CITED = 'Most Cited'
SORT_MOST_RECENT = 'Most Recent'

MAX_TITLE_LENGTH = 200
MAX_QUERY_LENGTH = 60

SEED = 19700101

log = logging.getLogger(__name__)


def reorder_publications(ids, pub_df):
    ids_order = {str(pid): index for index, pid in enumerate(ids)}
    sort_ord = np.argsort([ids_order[pid] for pid in pub_df['id']])
    return pub_df.iloc[sort_ord, :].reset_index(drop=True)


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


def is_doi(text):
    """
    Checks if text matches doi
    See https://www.crossref.org/blog/dois-and-matching-regular-expressions/
    :param text: text to check
    :return:
    """
    return re.search('10.\\d{4,9}/[-._;()/:a-z0-9A-Z]+', text.lower())


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


def contrast_color(rgb):
    r, g, b = rgb.r, rgb.g, rgb.b
    """
    Light foreground for dark background and vice verse.
    Idea Taken from https://stackoverflow.com/a/1855903/418358
    """
    # Counting the perceptive luminance - human eye favors green color...
    if 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5:
        return RGB(0, 0, 0)
    else:
        return RGB(255, 255, 255)


def color_to_rgb(v):
    return RGB(*[int(c * 255) for c in v[:3]])


def factor_colors(factors):
    cmap = factors_colormap(len(factors))
    palette = [color_to_rgb(cmap(i)).to_hex() for i in range(len(factors))]
    colors = factor_cmap('id', palette=palette, factors=factors)
    return colors


def topics_palette_rgb(df):
    n_comps = len(set(df['comp']))
    cmap = factors_colormap(n_comps)
    return dict((i, color_to_rgb(cmap(i))) for i in range(n_comps))


def factors_colormap(n):
    if n <= 10:
        return plt.cm.get_cmap('tab10', n)
    if n <= 20:
        return plt.cm.get_cmap('tab20', n)
    else:
        return plt.cm.get_cmap('nipy_spectral', n)


def topics_palette(df):
    return dict((k, v.to_hex()) for k, v in topics_palette_rgb(df).items())


def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0 or unit == 'GB':
            break
        size /= 1024.0
    return f'{int(size)} {unit}'
