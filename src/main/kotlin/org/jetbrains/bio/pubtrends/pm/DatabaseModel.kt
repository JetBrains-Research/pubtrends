package org.jetbrains.bio.pubtrends.pm

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.exposed.sql.Table
import org.postgresql.util.PGobject

internal const val PUBLICATION_MAX_TITLE_LENGTH = 1023

internal val jsonMapper = ObjectMapper()

class PGEnum<T : Enum<T>>(enumTypeName: String, enumValue: T?) : PGobject() {
    init {
        value = enumValue?.name
        type = enumTypeName
    }
}

enum class PublicationType {
    ClinicalTrial,
    Dataset,
    TechnicalReport,
    Review,
    Article
}

object PMPublications : Table() {
    val pmid = integer("pmid").primaryKey()
    val year = integer("year").nullable()
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
        index(true, pmidOut, pmidIn)
    }
}