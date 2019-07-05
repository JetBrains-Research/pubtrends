package org.jetbrains.bio.pubtrends.crawler

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.exposed.sql.Table

internal const val PUBLICATION_MAX_TITLE_LENGTH = 1023
internal const val KEYWORD_MAX_LENGTH = 80

internal val jsonMapper = ObjectMapper()

object PMPublications : Table() {
    val pmid = integer("pmid").primaryKey()
    val year = integer("year").nullable()
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()
    val type = text("type").nullable()
    val doi = text("doi").nullable()
    val aux = jsonb("aux", ArticleAuxInfo::class.java, jsonMapper)
}

object PMCitations : Table() {
    val pmidOut = integer("pmid_out")
    val pmidIn = integer("pmid_in")

    init {
        index(true, pmidOut, pmidIn)
    }
}

object PMKeywords : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val keyword = varchar("keyword", KEYWORD_MAX_LENGTH)

    init {
        index(true, keyword)
    }
}

object PMKeywordsPublications : Table() {
    val pmid = integer("pmid")
    val keywordId = integer("keyword_id")
    init {
        index(true, pmid, keywordId)
    }
}

object PMDatabanks : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val name = text("name")
    val accessionNumber = text("accession_number")

    init {
        index(true, name, accessionNumber)
    }
}

object PMDatabanksPublications : Table() {
    val pmid = integer("pmid")
    val databankId = integer("databank_id")

    init {
        index(true, pmid, databankId)
    }
}

object PMMeSH : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val meshId = text("mesh_id")
    val name = text("name")
    val type = text("type").nullable()
}

object PMMeSHPublications : Table() {
    val pmid = integer("pmid")
    val meshId = integer("mesh_id")
}