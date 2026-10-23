package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.pm.AuxInfo
import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.exposed.v1.core.Table

/**
 * ORM mapping for [PubmedArticle]
 */
object PMPublications : Table() {
    val pmid = integer("pmid")
    val year = integer("year")
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()

    val type = customEnumeration("type", "PublicationType",
        { value -> PublicationType.valueOf(value as String) }, { PGEnum("PublicationType", it) })

    val keywords = text("keywords").nullable()
    val mesh = text("mesh").nullable()
    val doi = text("doi").nullable()
    val aux = jsonb("aux", AuxInfo::class.java, jsonMapper)

    override val primaryKey = PrimaryKey(pmid)

    init {
        index(false, doi)
    }
}

object PMCitations : Table() {
    val pmidOut = integer("pmid_out")
    val pmidIn = integer("pmid_in")

    init {
        index(false, pmidIn)
        index(false, pmidOut)
    }
}
