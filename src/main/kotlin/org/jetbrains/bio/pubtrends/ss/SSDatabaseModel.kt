package org.jetbrains.bio.pubtrends.ss

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.bio.pubtrends.PGEnum
import org.jetbrains.bio.pubtrends.jsonb
import org.jetbrains.bio.pubtrends.ss.SSPublications.crc32id
import org.jetbrains.bio.pubtrends.ss.SSPublications.keywords
import org.jetbrains.bio.pubtrends.ss.SSPublications.ssid
import org.jetbrains.exposed.sql.Table

internal const val MAX_ID_LENGTH = 40
internal const val MAX_DOI_LENGTH = 100
internal const val PUBLICATION_MAX_TITLE_LENGTH = 1023

internal val jsonMapper = ObjectMapper()

/**
 * Table for storing Semantic Scholar Publications.
 *
 * @property ssid hex identifier produced by Semantic Scholar
 * @property crc32id ssid encoded by crc32, non-unique identifier that is used for the table index
 * @property keywords comma separated keywords
 */
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


enum class PublicationSource {
    Nature,
    Arxiv
}

object SSCitations : Table() {
    val id_out = varchar("id_out", MAX_ID_LENGTH)
    val id_in = varchar("id_in", MAX_ID_LENGTH)
}
