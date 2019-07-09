package org.jetbrains.bio.pubtrends.ssprocessing

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.bio.pubtrends.crawler.PUBLICATION_MAX_TITLE_LENGTH
import org.jetbrains.exposed.sql.Table
import org.postgresql.util.PGobject

internal const val MAX_ID_LENGTH = 40
internal const val MAX_DOI_LENGTH = 100

internal val jsonMapper = ObjectMapper()

object SSPublications : Table() {
    val ssid = varchar("ssid", MAX_ID_LENGTH)
    val crc32id = integer("crc32id")
    val pmid = integer("pmid").nullable()
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()
    val keywords = text("keywords").nullable()
    val year = integer("year").nullable()
    val doi = varchar("doi", MAX_DOI_LENGTH).nullable()
    val aux = jsonb("aux", ArticleAuxInfo::class.java, jsonMapper)

    val sourceEnum = customEnumeration("source", "Source",
            { value -> PublicationSource.valueOf(value as String) }, { PGEnum("Source", it) }).nullable()

    init {
        index(false, crc32id)
    }
}


enum class PublicationSource(source: String) {
    Nature("nature"),
    Arxiv("arxiv"),
}

class PGEnum<T : Enum<T>>(enumTypeName: String, enumValue: T?) : PGobject() {
    init {
        value = enumValue?.name
        type = enumTypeName
    }
}

object SSCitations : Table() {
    val id_out = varchar("id_out", MAX_ID_LENGTH) // from
    val id_in = varchar("id_in", MAX_ID_LENGTH) // to

//    init {
//        index(true, id_in, id_out)
//    }
}
