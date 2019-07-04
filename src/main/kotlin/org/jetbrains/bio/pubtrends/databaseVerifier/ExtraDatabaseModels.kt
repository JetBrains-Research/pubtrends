package org.jetbrains.bio.pubtrends.databaseVerifier

import org.jetbrains.exposed.sql.Table

internal const val MAX_ID_LENGTH = 100

object SemanticScholarCitations : Table() {
    val idCiting = varchar("id_citing", MAX_ID_LENGTH) // from
    val idCited = varchar("id_cited", MAX_ID_LENGTH) // to

    init {
        index(true, idCited, idCiting)
    }
}


object IdMatch : Table() {
    val pmid = integer("pmid").uniqueIndex()
    val ssid = varchar("ssid", MAX_ID_LENGTH).primaryKey()
}

object PmidCitationsFromSS : Table() {
    val pmidCiting = integer("pmid_citing")
    val pmidCited = integer("pmid_cited")

    init {
        index(true, pmidCiting, pmidCited)
    }
}