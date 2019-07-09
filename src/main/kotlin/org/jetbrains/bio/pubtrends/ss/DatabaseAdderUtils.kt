package org.jetbrains.bio.pubtrends.ss

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.batchInsert
import org.jetbrains.exposed.sql.transactions.transaction

object DatabaseAdderUtils {
    private val logger = LogManager.getLogger(DatabaseAdderUtils::class)

    fun addArticles(articles: MutableList<SemanticScholarArticle>) {
        val citationsList = articles.map { it.citationList.distinct().map { cit -> it.ssid to cit } }.flatten()

        transaction {
            // addLogger(StdOutSqlLogger)

            SSPublications.batchInsert(articles, ignore = true) { article ->
                this[SSPublications.ssid] = article.ssid
                this[SSPublications.pmid] = article.pmid
                this[SSPublications.abstract] = article.abstract
                this[SSPublications.keywords] = article.keywords
                this[SSPublications.title] = article.title
                this[SSPublications.year] = article.year
                this[SSPublications.doi] = article.doi
                this[SSPublications.sourceEnum] = article.source
                this[SSPublications.aux] = article.aux
                this[SSPublications.crc32id] = article.crc32id
            }

            SSCitations.batchInsert(citationsList, ignore = true) { citation ->
                this[SSCitations.id_out] = citation.first
                this[SSCitations.id_in] = citation.second
            }
        }
    }
}