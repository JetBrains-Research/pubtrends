package org.jetbrains.bio.pubtrends.db

enum class ArxivNeo4JDBLabels(val str: String) {
    PUBLICATION("Publication"),
    AUTHOR("Author"),
    AFFILIATION("Affiliation"),
    JOURNAL("Journal"),
    MISSING_PUBLICATION("MissingPublication"),
    AUTHORED("AUTHORED_BY"),
    WORKS("WORKS"),
    CITES("CITES"),
    CITED_BY("CITED_BY"),
    PUBLISHED_IN("PUBLISHED_IN"),
    ARXIV_LBL("Arxiv")
}