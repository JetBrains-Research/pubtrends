import json
from io import StringIO

import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from scipy.sparse import csr_matrix


class AnalysisData:
    def __init__(self, search_query, search_ids,
                 source, sort, limit, noreviews, min_year, max_year,
                 df, cit_df, cocit_grouped_df, bibliographic_coupling_df,
                 top_cited_df, max_gain_df, max_rel_gain_df,
                 corpus, corpus_tokens, corpus_counts,
                 papers_graph, papers_embeddings,
                 dendrogram,
                 author_stats, journal_stats, numbers_df):
        self.search_query = search_query  # Initial query for analysis
        self.search_ids = search_ids  # Initial ids for analysis
        self.source = source
        self.sort = sort
        self.limit = limit
        self.noreviews = noreviews
        self.min_year = min_year
        self.max_year = max_year
        self.df = df
        self.cit_df = cit_df
        self.cocit_grouped_df = cocit_grouped_df
        self.bibliographic_coupling_df = bibliographic_coupling_df
        self.top_cited_df = top_cited_df
        self.max_gain_df = max_gain_df
        self.max_rel_gain_df = max_rel_gain_df
        self.corpus = corpus
        self.corpus_tokens = corpus_tokens
        self.corpus_counts = corpus_counts
        self.papers_graph = papers_graph
        self.papers_embeddings = papers_embeddings
        self.dendrogram = dendrogram
        self.author_stats = author_stats
        self.journal_stats = journal_stats
        self.numbers_df = numbers_df

    def to_json(self):
        """
        Dump valuable fields to JSON-serializable dict.
        """
        csm_json = json.dumps(dict(
            data=self.corpus_counts.data.tolist(),
            indices=self.corpus_counts.nonzero()[0].tolist(),
            indptr=self.corpus_counts.nonzero()[1].tolist())
        )

        return dict(
            search_query=self.search_query,
            search_ids=self.search_ids,
            source=self.source,
            sort=self.sort,
            limit=self.limit,
            noreviews=self.noreviews,
            min_year=self.min_year,
            max_year=self.max_year,
            df=self.df.to_json(),
            cit_df=self.cit_df.to_json(),
            cocit_grouped_df=self.cocit_grouped_df.to_json(),
            bibliographic_coupling_df=self.bibliographic_coupling_df.to_json(),
            top_cited_df=self.top_cited_df.to_json(),
            max_gain_df=self.max_gain_df.to_json(),
            max_rel_gain_df=self.max_rel_gain_df.to_json(),
            dendrogram=self.dendrogram.tolist() if self.dendrogram is not None else None,
            corpus=self.corpus,
            corpus_tokens=self.corpus_tokens,
            corpus_counts=csm_json,
            papers_graph=json_graph.node_link_data(self.papers_graph),
            papers_embeddings=self.papers_embeddings.tolist(),
            author_stats=self.author_stats.to_json() if self.author_stats is not None else None,
            journal_stats=self.journal_stats.to_json() if self.journal_stats is not None else None,
            numbers_df=self.numbers_df.to_json() if self.numbers_df is not None else None,
        )

    @staticmethod
    def from_json(fields) -> 'AnalysisData':
        """
        Load from JSON-serializable dict.
        """
        search_ids = fields['search_ids']
        search_query = fields['search_query']
        source = fields['source']
        sort = fields['sort']
        limit = fields['limit']
        noreviews = fields['noreviews']
        min_year = fields['min_year']
        max_year = fields['max_year']
        # Restore main dataframe
        df = pd.read_json(StringIO(fields['df']))
        df['id'] = df['id'].apply(str)
        mapping = {}
        for col in df.columns:
            try:
                mapping[col] = int(col)
            except ValueError:
                mapping[col] = col
        df = df.rename(columns=mapping)

        cit_df = pd.read_json(StringIO(fields['cit_df']))
        cit_df['id_in'] = cit_df['id_in'].astype(str)
        cit_df['id_out'] = cit_df['id_out'].astype(str)

        cocit_grouped_df = pd.read_json(StringIO(fields['cocit_grouped_df']))
        cocit_grouped_df['cited_1'] = cocit_grouped_df['cited_1'].astype(str)
        cocit_grouped_df['cited_2'] = cocit_grouped_df['cited_2'].astype(str)

        bibliographic_coupling_df = pd.read_json(StringIO(fields['bibliographic_coupling_df']))
        bibliographic_coupling_df['citing_1'] = bibliographic_coupling_df['citing_1'].astype(str)
        bibliographic_coupling_df['citing_2'] = bibliographic_coupling_df['citing_2'].astype(str)

        top_cited_df = pd.read_json(StringIO(fields['top_cited_df']))
        top_cited_df['id'] = top_cited_df['id'].apply(str)
        max_gain_df = pd.read_json(StringIO(fields['max_gain_df']))
        max_gain_df['id'] = max_gain_df['id'].apply(str)
        max_rel_gain_df = pd.read_json(StringIO(fields['max_rel_gain_df']))
        max_rel_gain_df['id'] = max_rel_gain_df['id'].apply(str)

        # Corpus information
        corpus = fields['corpus']
        corpus_tokens = fields['corpus_tokens']
        corpus_counts = json.loads(fields['corpus_counts'])
        corpus_counts = csr_matrix((corpus_counts['data'], (corpus_counts['indices'], corpus_counts['indptr'])))


        # Restore citation and structure graphs
        papers_graph = json_graph.node_link_graph(fields['papers_graph'])

        # Restore original embeddings
        papers_embeddings = np.array(fields['papers_embeddings'])

        # Restore dendrogram
        dendrogram = fields['dendrogram']
        if dendrogram is not None:
            dendrogram = np.array(dendrogram)

        # Restore additional analysis
        author_stats = pd.read_json(StringIO(fields['author_stats'])) if fields['author_stats'] is not None else None
        journal_stats = pd.read_json(StringIO(fields['journal_stats'])) if fields['journal_stats'] is not None else None
        numbers_df = pd.read_json(StringIO(fields['numbers_df'])) if fields['numbers_df'] is not None else None

        return AnalysisData(
            search_query=search_query,
            search_ids=search_ids,
            source=source,
            sort=sort,
            limit=limit,
            noreviews=noreviews,
            min_year=min_year,
            max_year=max_year,
            df=df,
            cit_df=cit_df,
            cocit_grouped_df=cocit_grouped_df,
            bibliographic_coupling_df=bibliographic_coupling_df,
            top_cited_df=top_cited_df,
            max_gain_df=max_gain_df,
            max_rel_gain_df=max_rel_gain_df,
            corpus=corpus,
            corpus_tokens=corpus_tokens,
            corpus_counts=corpus_counts,
            papers_embeddings=papers_embeddings,
            papers_graph=papers_graph,
            dendrogram=dendrogram,
            author_stats=author_stats,
            journal_stats=journal_stats,
            numbers_df=numbers_df,
        )
