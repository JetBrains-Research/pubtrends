package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.pm.AuxInfo
import org.jetbrains.exposed.sql.Table

internal const val ARXIV_ID_MAX_LENGTH = 15

/**
 * Table for storing information about arXiv preprints.
 */
object ArxivPreprints : Table() {
    val arxivId = varchar("arxiv_id", ARXIV_ID_MAX_LENGTH).primaryKey()
    val date = date("date").nullable()
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()
    val doi = text("doi").nullable()
    val aux = jsonb("aux", AuxInfo::class.java, jsonMapper)

    init {
        index(false, doi)
    }
}

object ArxivCitations : Table() {
    val arxivIdOut = integer("arxiv_id_out")
    val arxivIdIn = integer("arxiv_id_in")

    init {
        index(true, arxivIdIn, arxivIdOut)
        index(false, arxivIdOut)
    }
}

object ArxivSSCitations : Table() {
    val arxivIdOut = varchar("arxiv_id_out", ARXIV_ID_MAX_LENGTH)
    val ssidIn = varchar("ssid_in", MAX_ID_LENGTH)
    val crc32idIn = integer("crc32id_in")

    init {
        ArxivCitations.index(true, crc32idIn, arxivIdOut)
        ArxivCitations.index(false, arxivIdOut)
    }
}