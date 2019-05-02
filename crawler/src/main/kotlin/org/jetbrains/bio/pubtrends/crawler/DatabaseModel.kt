package org.jetbrains.bio.pubtrends.crawler

import org.jetbrains.exposed.sql.Table

internal const val PUBLICATION_MAX_TITLE_LENGTH = 1023
internal const val KEYWORD_MAX_LENGTH = 80

object Publications : Table() {
    val pmid = integer("pmid").primaryKey()
    val year = integer("year").nullable()
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()
}

object Citations : Table() {
    val pmidCiting = integer("pmid_citing")
    val pmidCited = integer("pmid_cited")

    init {
        index(true, pmidCiting, pmidCited)
    }
}

object Keywords : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val keyword = varchar("keyword", KEYWORD_MAX_LENGTH)

    init {
        index(true, keyword)
    }
}

object KeywordsPublications : Table() {
    val pmid = integer("pmid")
    val keywordId = integer("keyword_id")
    init {
        index(true, pmid, keywordId)
    }
}