package org.jetbrains.bio.pubtrends.crawler

import org.jetbrains.exposed.sql.Table

object Publications : Table() {
    val pmid = integer("pmid").primaryKey()
    val year = integer("year").nullable()
    val title = varchar("title", 1023)
    val abstract = text("abstract")

    init {
        index(true, title)
        index(true, abstract)
    }
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
    val keyword = varchar("keyword", 80)

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