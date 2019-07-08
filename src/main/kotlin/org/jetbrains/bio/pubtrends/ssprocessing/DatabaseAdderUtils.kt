package org.jetbrains.bio.pubtrends.ssprocessing

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.crawler.batchInsertOnDuplicateKeyUpdate
import org.jetbrains.exposed.sql.batchInsert
import org.jetbrains.exposed.sql.transactions.transaction

object DatabaseAdderUtils {
    private val logger = LogManager.getLogger(DatabaseAdderUtils::class)

    fun addArticles(articles: MutableList<SemanticScholarArticle>) {
        val keywordList = articles.map { it.keywordList }.flatten().distinct()
        val citationsList = articles.map { it.citationList.distinct().map { cit -> it.ssid to cit } }.flatten()
        val keywordsForArticle = articles.map {
            it.keywordList.map { keywords -> it.ssid to keywords }
        }.flatten()

        transaction {
            // addLogger(StdOutSqlLogger)

            SSPublications.batchInsert(articles, ignore = true) { article ->
                this[SSPublications.ssid] = article.ssid
                this[SSPublications.pmid] = article.pmid
                this[SSPublications.abstract] = article.abstract
                this[SSPublications.title] = article.title
                this[SSPublications.year] = article.year
                this[SSPublications.doi] = article.doi
                this[SSPublications.sourceEnum] = article.source
                this[SSPublications.aux] = article.aux
            }

            val keywordsId = SSKeywords.batchInsertOnDuplicateKeyUpdate(
                    keywordList,
                    SSKeywords.keyword,
                    listOf(SSKeywords.keyword)) { batch, kw ->
                batch[SSKeywords.keyword] = kw
            }

            val keywordIdMap = keywordList.zip(keywordsId).toMap()

            SSKeywordsPublications.batchInsert(keywordsForArticle) { (p_id, keyword) ->
                this[SSKeywordsPublications.sspid] = p_id
                this[SSKeywordsPublications.sskid] = keywordIdMap[keyword]!!
            }

            SSCitations.batchInsert(citationsList, ignore = true) { citation ->
                this[SSCitations.id_out] = citation.first
                this[SSCitations.id_in] = citation.second
            }
        }
    }
}