package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.db.SSPublications.crc32id
import org.jetbrains.bio.pubtrends.db.SSPublications.keywords
import org.jetbrains.bio.pubtrends.db.SSPublications.ssid
import org.jetbrains.bio.pubtrends.ss.ArticleAuxInfo
import org.jetbrains.exposed.sql.Table

internal const val MAX_ID_LENGTH = 40
internal const val MAX_DOI_LENGTH = 100

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

    init {
        index(false, crc32id)
    }
}


object SSCitations : Table() {
    val id_out = varchar("id_out", MAX_ID_LENGTH)
    val id_in = varchar("id_in", MAX_ID_LENGTH)

    val crc32id_out = integer("crc32id_out")
    val crc32id_in = integer("crc32id_in")

    init {
        index(false, crc32id_in, crc32id_out)
        index(false, crc32id_out)
    }
}
