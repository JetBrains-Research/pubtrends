import json
import logging
from itertools import chain

from pysrc.app.reports import preprocess_string
from pysrc.config import PubtrendsConfig, VISUALIZATION_GRAPH_EDGES, TOPIC_DESCRIPTION_WORDS
from pysrc.services.embeddings_service import is_texts_embeddings_available
from pysrc.papers.analysis.graph import sparse_graph
from pysrc.papers.analysis.text import get_frequent_tokens
from pysrc.papers.analysis.topics import get_topics_description
from pysrc.papers.data import AnalysisData
from pysrc.papers.db.loaders import Loaders
from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor
from pysrc.papers.plot.plotter import Plotter, components_list, topics_info_and_word_cloud
from pysrc.papers.utils import trim_query, MAX_TITLE_LENGTH, topics_palette, MAX_QUERY_LENGTH, trim, PAPER_ANALYSIS_TYPE

logger = logging.getLogger(__name__)


def prepare_result_data(config: PubtrendsConfig, data: AnalysisData):
    plotter = Plotter(config, data)
    freq_kwds = get_frequent_tokens(chain(*chain(*data.corpus)))
    word_cloud = plotter._papers_word_cloud(freq_kwds)
    export_name = preprocess_string(data.search_query)
    result = dict(
        topics_analyzed=False,
        n_papers=len(data.df),
        n_topics=len(set(data.df['comp'])),
        export_name=export_name,
        top_cited_papers=components_list(plotter.plot_top_cited_papers()),
        most_cited_per_year_papers=components_list(plotter.plot_most_cited_per_year_papers()),
        fastest_growth_per_year_papers=components_list(plotter.plot_fastest_growth_per_year_papers()),
        papers_stats=components_list(plotter.plot_papers_by_year()),
        papers_word_cloud=PlotPreprocessor.word_cloud_prepare(word_cloud),
    )

    keywords_frequencies = plotter.plot_keywords_frequencies(freq_kwds)
    if keywords_frequencies is not None:
        result['keywords_frequencies'] = components_list(keywords_frequencies)

    if data.papers_graph.nodes():
        result.update(dict(
            topics_analyzed=True,
            topic_years_distribution=components_list(plotter.plot_topic_years_distribution()),
            topics_info_and_word_cloud=topics_info_and_word_cloud(plotter),
            component_sizes=PlotPreprocessor.component_sizes(data.df),
            papers_graph=components_list(plotter.plot_papers_graph())
        ))

        topics_hierarchy_with_keywords = plotter.topics_hierarchy_with_keywords()
        if topics_hierarchy_with_keywords:
            result['topics_hierarchy_with_keywords'] = components_list(topics_hierarchy_with_keywords)


    result['feature_authors_enabled'] = config.feature_authors_enabled
    if config.feature_authors_enabled:
        result['author_statistics'] = plotter.author_statistics()

    result['feature_journals_enabled'] = config.feature_journals_enabled
    if config.feature_journals_enabled:
        result['journal_statistics'] = plotter.journal_statistics()

    result['feature_numbers_enabled'] = config.feature_numbers_enabled
    if config.feature_numbers_enabled:
        url_prefix = Loaders.get_url_prefix(data.source)
        if data.numbers_df is not None:
            result['numbers'] = [
                (row['id'], url_prefix + str(row['id']), trim(row['title'], MAX_TITLE_LENGTH), row['numbers'])
                for _, row in data.numbers_df.iterrows()
            ]

    result['feature_questions_enabled'] = config.feature_questions_enabled and is_texts_embeddings_available()

    return result

def prepare_search_string(topic, word, author, journal, papers_list) -> tuple[int, str]:
    search_string = ''
    if topic is not None:
        search_string += f'topic: {topic}'
        comp = int(topic) - 1  # Component was exposed so it was 1-based
    else:
        comp = None

    if word is not None:
        search_string += f'word: {word}'

    if author is not None:
        search_string += f'author: {author}'

    if journal is not None:
        search_string += f'journal: {journal}'

    if papers_list == 'top':
        search_string += 'Top Papers'
    if papers_list == 'year':
        search_string += 'Papers of the Year'
    if papers_list == 'hot':
        search_string += 'Hot Papers'

    return comp, search_string


def prepare_papers_data(
        data: AnalysisData,
        comp=None, word=None, author=None, journal=None, papers_list=None
):
    df = data.df
    top_cited_papers = list(data.top_cited_df['id'])
    max_gain_papers = list(data.max_gain_df['id'])
    max_rel_gain_papers = list(data.max_rel_gain_df['id'])
    url_prefix = Loaders.get_url_prefix(data.source)
    # Filter by component
    if comp is not None:
        df = df[df['comp'].astype(int) == comp]
    # Filter by word
    if word is not None:
        df = df[(df['title'].str.contains(word, case=False)) |
                (df['abstract'].str.contains(word, case=False)) |
                (df['mesh'].str.contains(word, case=False)) |
                (df['keywords'].str.contains(word, case=False))]
    # Filter by author
    if author is not None:
        # Check if string was trimmed
        if author.endswith('...'):
            author = author[:-3]
            df = df[[any([a.startswith(author) for a in authors]) for authors in df['authors']]]
        else:
            df = df[[author in authors for authors in df['authors']]]

    # Filter by journal
    if journal is not None:
        # Check if string was trimmed
        if journal.endswith('...'):
            journal = journal[:-3]
            df = df[[j.startswith(journal) for j in df['journal']]]
        else:
            df = df[df['journal'] == journal]

    if papers_list == 'top':
        df = df[[pid in top_cited_papers for pid in df['id']]]
    if papers_list == 'year':
        df = df[[pid in max_gain_papers for pid in df['id']]]
    if papers_list == 'hot':
        df = df[[pid in max_rel_gain_papers for pid in df['id']]]

    result = []
    for _, row in df.iterrows():
        pid, title, authors, journal, year, total, doi, topic = \
            row['id'], row['title'], row['authors'], row['journal'], \
                row['year'], int(row['total']), str(row['doi']), int(row['comp'] + 1)
        if doi == 'None' or doi == 'nan':
            doi = ''
        # Don't trim or cut anything here, because this information can be exported
        result.append(
            (pid, title, authors, url_prefix + pid if url_prefix else None, journal, year, total, doi, topic)
        )
    return result


def prepare_paper_data(data: AnalysisData, pid):
    logger.debug('Extracting data for the current paper')
    source = data.source
    url_prefix = Loaders.get_url_prefix(source)
    # Use pid if is given, or for PAPER_ANALYSIS_TYPE use the seed paper, otherwise show the very first paper
    if not pid:
        pid = data.search_ids[0] if data.analysis_type == PAPER_ANALYSIS_TYPE and data.search_ids else data.df['id'].values[0]
    sel = data.df[data.df['id'] == pid]
    title = sel['title'].values[0]
    authors = sel['authors'].values[0]
    journal = sel['journal'].values[0]
    year = sel['year'].values[0]
    topic = sel['comp'].values[0] + 1
    doi = str(sel['doi'].values[0])
    mesh = str(sel['mesh'].values[0])
    keywords = str(sel['keywords'].values[0])
    if doi == 'None' or doi == 'nan':
        doi = ''

    topics_description = get_topics_description(
        data.df,
        data.corpus, data.corpus_tokens, data.corpus_counts,
        n_words=TOPIC_DESCRIPTION_WORDS,
    )
    topic_tags = ','.join(w[0] for w in topics_description[topic - 1][:TOPIC_DESCRIPTION_WORDS])

    logger.debug('Computing most cited prior and derivative papers')
    derivative_papers = PlotPreprocessor.get_top_papers_id_title_year_cited_topic(
        data.cit_df.loc[data.cit_df['id_in'] == pid]['id_out'], data.df
    )
    prior_papers = PlotPreprocessor.get_top_papers_id_title_year_cited_topic(
        data.cit_df.loc[data.cit_df['id_out'] == pid]['id_in'], data.df
    )

    logger.debug('Computing most similar papers')
    if len(data.df) > 1:
        graph = data.papers_graph
        similar_papers = []
        neighbors_data = sorted(list([(x, graph.get_edge_data(pid, x)) for x in graph.neighbors(pid)]),
                                key=lambda x: x[1]['similarity'], reverse=True)
        for v, d in neighbors_data[:50]:
            similar_papers.append(
                (v,
                 data.df[data.df['id'] == v]['title'].values[0],
                 data.df[data.df['id'] == v]['year'].values[0],
                 data.df[data.df['id'] == v]['total'].values[0],
                 d['similarity'],
                 data.df[data.df['id'] == v]['comp'].values[0] + 1
                 )
            )
    else:
        similar_papers = None

    result = dict(title=title,
                  query=trim_query(data.search_query),
                  source=source,
                  sort=data.sort or '',
                  limit=data.limit,
                  authors=authors,
                  journal=journal,
                  year=year,
                  doi=doi,
                  mesh=mesh,
                  url=url_prefix + pid,
                  topic=topic,
                  keywords=keywords,
                  n_papers=len(data.df),
                  topics_palette=topics_palette(data.df),
                  topic_tags=topic_tags,
                  citation_dynamics=components_list(Plotter._plot_paper_citations_per_year(data.df, str(pid))))

    if similar_papers:
        result['similar_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid,
                                     year, cited, f'{similarity:.3f}', topic)
                                    for pid, title, year, cited, similarity, topic in similar_papers]

    abstract = sel['abstract'].values[0]
    if abstract != '':
        result['abstract'] = abstract

    if len(prior_papers) > 0:
        result['prior_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year, cited, topic + 1)
                                  for pid, title, year, cited, topic in prior_papers]

    if len(derivative_papers) > 0:
        result['derivative_papers'] = [(pid, trim(title, MAX_TITLE_LENGTH), url_prefix + pid, year, cited, topic + 1)
                                       for pid, title, year, cited, topic in derivative_papers]

    if len(data.df) > 1:
        logger.debug('Compute topics description')
        topics_description = get_topics_description(
            data.df,
            data.corpus, data.corpus_tokens, data.corpus_counts,
            n_words=TOPIC_DESCRIPTION_WORDS,
        )
        logger.debug('Prepare sparse graph to visualize with reduced number of edges')
        visualize_graph = sparse_graph(data.papers_graph, VISUALIZATION_GRAPH_EDGES)
        result['papers_graph'] = components_list(
            Plotter._plot_papers_graph(
                data.search_ids, source, visualize_graph, data.df, TOPIC_DESCRIPTION_WORDS,
                shown_pid=pid, topics_tags=topics_description
            ))

    return result


def prepare_graph_data(data: AnalysisData, shown_id=None):
    logger.debug('Extracting graph data')
    topics_description = get_topics_description(
        data.df,
        data.corpus, data.corpus_tokens, data.corpus_counts,
        n_words=TOPIC_DESCRIPTION_WORDS,
    )
    topics_tags = {
        comp: ','.join([w[0] for w in topics_description[comp]]) for comp in sorted(set(data.df['comp']))
    }
    logger.debug('Computing sparse graph')
    visualize_graph = sparse_graph(data.papers_graph, VISUALIZATION_GRAPH_EDGES)
    graph_cs = PlotPreprocessor.dump_similarity_graph_cytoscape(
        data.df, visualize_graph
    )
    return dict(
        query=trim_query(data.search_query),
        source=data.source,
        # Don't highlight any search_ids if all the papers are there
        search_ids=json.dumps(data.search_ids if data.search_ids and len(data.search_ids) < len(data.df) else []),
        shown_id=shown_id,
        limit=data.limit,
        sort=data.sort or '',
        topics_palette_json=json.dumps(topics_palette(data.df)),
        topics_tags_json=json.dumps(topics_tags),
        graph_cytoscape_json=json.dumps(graph_cs)
    )
