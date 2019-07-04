package org.jetbrains.bio.pubtrends.databaseVerifier

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.transactions.transaction

class DatabaseAdderUtils {
    companion object {
        private val logger = LogManager.getLogger(DatabaseAdderUtils::class)
    }

    fun addArticle(hashId: String, pmid: Int, citations: List<String>) {
        transaction {
            addLogger(StdOutSqlLogger)

            IdMatch.insertIgnore {
                it[this.pmid] = pmid
                it[ssid] = hashId
            }


            if (citations.isNotEmpty()) {
                SemanticScholarCitations.batchInsert(citations, ignore = true) { citation ->
                    this[SemanticScholarCitations.idCiting] = hashId
                    this[SemanticScholarCitations.idCited] = citation
                }
            }
        }
    }


    fun addArticles(articles: MutableList<SemanticScholarArticle>) {
        val citationsForArticle = articles.map { it.citations.toSet().map { cit -> it.id to cit } }.flatten()

        transaction {
//            addLogger(StdOutSqlLogger)

            IdMatch.batchInsert(articles, ignore = true) { article ->
                this[IdMatch.pmid] = article.pmid
                this[IdMatch.ssid] = article.id
            }
            SemanticScholarCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[SemanticScholarCitations.idCiting] = citation.first
                this[SemanticScholarCitations.idCited] = citation.second
            }

        }
    }

    fun addPmidCitations() {
        // here will be join query to fill PmidCitationsFromSS table
        // can't understand what "exposed" can do for this
        val sqlAddStatement = """
            INSERT INTO PmidCitationsFromSS
            SELECT
            t1.pmid AS pmid_citing,
            t2.pmid AS pmid_cited
            FROM SemanticScholarCitations
            INNER JOIN IdMatch AS t1
            ON t1.ssid = SemanticScholarCitations.id_citing
            INNER JOIN IdMatch AS t2
            ON t2.ssid = SemanticScholarCitations.id_cited;
            """
        logger.info("To add all needed data to database please run \"psql pubmed\" and the following command:")
        logger.info(sqlAddStatement)
    }

}