package org.jetbrains.bio.pubtrends.ssprocessing

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.bio.pubtrends.crawler.PUBLICATION_MAX_TITLE_LENGTH
import org.jetbrains.exposed.sql.Table
import org.postgresql.util.PGobject

internal const val MAX_ID_LENGTH = 60
internal const val MAX_DOI_LENGTH = 100
internal const val SS_KEYWORD_MAX_LENGTH = 30

internal val jsonMapper = ObjectMapper()

object SSPublications : Table() {
    val id = integer("id").autoIncrement()
    val ssid = varchar("ssid", MAX_ID_LENGTH)
    val pmid = integer("pmid").nullable()
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()
    val year = integer("year").nullable()
    val doi = varchar("doi", MAX_DOI_LENGTH).nullable()
    val aux = jsonb("aux", ArticleAuxInfo::class.java, jsonMapper)

    val sourceEnum = customEnumeration("source", "Source",
            {value -> PublicationSource.valueOf(value as String)}, { PGEnum("Source", it)}).nullable()
}


enum class PublicationSource(source: String) {
    Nature("nature"),
    Arxiv("arxiv"),
}

class PGEnum<T:Enum<T>>(enumTypeName: String, enumValue: T?) : PGobject() {
    init {
        value = enumValue?.name
        type = enumTypeName
    }
}

object SSCitations : Table() {
    val id_out = integer("id_out") // from
    val id_in = integer("id_in") // to

    init {
        index(true, id_in, id_out)
    }
}


object SSKeywords : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val keyword = varchar("keyword", SS_KEYWORD_MAX_LENGTH).uniqueIndex()

//    init {
//        index(true, keyword)
//    }
}

object SSKeywordsPublications : Table() {
    val sspid = integer("sspid")
    val sskid = integer("sskid")
//    init {
//        index(true, pmid, keywordId)
//    }
}