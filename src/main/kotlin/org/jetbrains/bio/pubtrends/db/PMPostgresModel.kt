package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.pm.ArticleAuxInfo
import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.exposed.sql.Table

object PMPublications : Table() {
    val pmid = integer("pmid").primaryKey()
    val date = date("date").nullable()
    val title = varchar("title", PUBLICATION_MAX_TITLE_LENGTH)
    val abstract = text("abstract").nullable()

    val type = customEnumeration("type", "PublicationType",
            { value -> PublicationType.valueOf(value as String) }, { PGEnum("PublicationType", it) })

    val keywords = text("keywords").nullable()
    val mesh = text("mesh").nullable()
    val doi = text("doi").nullable()
    val aux = jsonb("aux", ArticleAuxInfo::class.java, jsonMapper)
}

object PMCitations : Table() {
    val pmidOut = integer("pmid_out")
    val pmidIn = integer("pmid_in")

    init {
        index(true, pmidIn, pmidOut)
        index(false, pmidOut)
    }
}