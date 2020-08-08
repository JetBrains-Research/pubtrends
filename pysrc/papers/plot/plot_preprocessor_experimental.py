import logging

import pandas as pd
from bokeh.models import ColumnDataSource, TableColumn

from pysrc.papers.plot.plot_preprocessor import PlotPreprocessor

logger = logging.getLogger(__name__)


class ExperimentalPlotPreprocessor(PlotPreprocessor):

    @staticmethod
    def topic_evolution_data(df, kwds, n_steps):
        def sort_nodes_key(node):
            y, c = node[0].split(' ')
            return int(y), -int(c)

        cols = df.columns[2:]
        pairs = list(zip(cols, cols[1:]))
        nodes = set()
        edges = []
        for now, then in pairs:
            nodes_now = [f'{now} {c}' for c in df[now].unique()]
            nodes_then = [f'{then} {c}' for c in df[then].unique()]

            inner = {node: 0 for node in nodes_then}
            changes = {node: inner.copy() for node in nodes_now}
            for pmid, comp in df.iterrows():
                c_now, c_then = comp[now], comp[then]
                changes[f'{now} {c_now}'][f'{then} {c_then}'] += 1

            for v in nodes_now:
                for u in nodes_then:
                    if changes[v][u] > 0:
                        edges.append((v, u, changes[v][u]))
                        nodes.add(v)
                        nodes.add(u)

        nodes_data = []
        for node in nodes:
            year, c = node.split(' ')
            if int(c) >= 0:
                if n_steps < 4:
                    label = f"{year} {', '.join(kwds[int(year)][int(c)][:5])}"
                else:
                    # Fix topic numbering to start with 1
                    label = f"{year} {int(c) + 1}"
            else:
                label = "TBD"
            nodes_data.append((node, label))
        nodes_data = sorted(nodes_data, key=sort_nodes_key, reverse=True)

        return edges, nodes_data

    @staticmethod
    def topic_evolution_keywords_data(kwds):
        years = []
        topics = []
        keywords = []
        for year, comps in kwds.items():
            for c, kwd in comps.items():
                if c >= 0:
                    years.append(year)
                    topics.append(c + 1)
                    keywords.append(', '.join(kwd))
        data = dict(
            years=years,
            topics=topics,
            keywords=keywords
        )
        source = ColumnDataSource(data)
        source.add(pd.RangeIndex(start=1, stop=len(keywords), step=1), 'index')
        columns = [
            TableColumn(field="index", title="#", width=20),
            TableColumn(field="years", title="Year", width=50),
            TableColumn(field="topics", title="Topic", width=50),
            TableColumn(field="keywords", title="Keywords", width=800),
        ]
        return columns, source
