package org.jetbrains.bio.pubtrends.ssprocessing

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.crawler.batchInsertOnDuplicateKeyUpdate
import org.jetbrains.exposed.sql.batchInsert
import org.jetbrains.exposed.sql.select
import org.jetbrains.exposed.sql.transactions.transaction

object DatabaseAdderUtils {
    private val logger = LogManager.getLogger(DatabaseAdderUtils::class)

    fun addArticles(articles: MutableList<SemanticScholarArticle>) {
        val keywordList = articles.map { it.keywordList }.flatten().distinct()

        transaction {
            // addLogger(StdOutSqlLogger)

            val articlesId = SSPublications.batchInsert(articles) { article ->
                this[SSPublications.ssid] = article.ssid
                this[SSPublications.pmid] = article.pmid
                this[SSPublications.abstract] = article.abstract
                this[SSPublications.title] = article.title
                this[SSPublications.year] = article.year
                this[SSPublications.doi] = article.doi
                this[SSPublications.sourceEnum] = article.source
                this[SSPublications.aux] = article.aux
            }

            val articlesIdMatch = articles.map { it.ssid }
                    .zip(articlesId.map { it[SSPublications.id].toString().toInt() }).toMap()

            val keywordsForArticle = articles.map {
                it.keywordList.toSet().map { keywords -> articlesIdMatch[it.ssid] to keywords }
            }.flatten()

            // keywordsForArticle example: [(23001, Metabolic Biotransformation), (23001, Renal Elimination), (23001, Sulfamethizole)]

            val keywordsId = SSKeywords.batchInsertOnDuplicateKeyUpdate(
                    keywordList,
                    SSKeywords.keyword,
                    listOf(SSKeywords.keyword)) { batch, kw ->
                batch[SSKeywords.keyword] = kw
            }

            val keywordIdMap = keywordList.zip(keywordsId).toMap()

            // keywordIdMap example: {Big data=752, Choose (action)=505, Clustering high-dimensional data=109904}

            SSKeywordsPublications.batchInsert(keywordsForArticle) { (p_id, keyword) ->
                this[SSKeywordsPublications.sspid] = p_id!!
                this[SSKeywordsPublications.sskid] = keywordIdMap[keyword]!!
            }
        }
    }

    fun addCitations(articles: MutableList<SemanticScholarArticle>) {
        val articlesIdOut = articles.map { it.ssid }
        val articlesIdIn = articles.map { it.citationList }.flatten()
        val articlesSet = articlesIdOut.union(articlesIdIn)

        transaction {
            // addLogger(StdOutSqlLogger)

            val articleIdMap = articlesSet.associateWith { ssid ->
                SSPublications.slice(SSPublications.id).select { SSPublications.ssid eq ssid }.single()[SSPublications.id]
            }

            val citationsForArticle = articles
                    .map { it.citationList.map { cit -> articleIdMap[it.ssid] to articleIdMap[cit] } }
                    .flatten()


            SSCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[SSCitations.id_out] = citation.first!!
                this[SSCitations.id_in] = citation.second!!
            }
        }
    }
}