package org.jetbrains.bio.pubtrends.databaseVerifier

import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.transactions.transaction


fun addArticle(hashId: String, pmid: Int, citations: List<String>) {
    transaction {
//        addLogger(StdOutSqlLogger)
        if (pmid != null && hashId != null) {
            idMatch.insertIgnore {
                it[this.pmid] = pmid
                it[ssid] = hashId }
        }

        if (citations.isNotEmpty()) {
            SemanticScholarCitations.batchInsert(citations, ignore = true) { citation ->
                this[SemanticScholarCitations.idCiting] = hashId
                this[SemanticScholarCitations.idCited] = citation
            }
        }
    }
}

fun addPmidCitations() {
    // here will be join query to fill PmidCitationsFromSS table

    // can't understand what "exposed" can do for this

//    idMatch.innerJoin(SemanticScholarCitations, {idMatch.ssid}, {SemanticScholarCitations.idCiting})
//
//    (idMatch innerJoin SemanticScholarCitations).slice(SemanticScholarCitations.idCiting, idMatch.ssid).
//            select {}.forEach {
//    }
}