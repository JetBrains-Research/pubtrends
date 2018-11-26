package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction

class DatabaseHandler() {
    companion object Log4jSqlLogger : SqlLogger {
        private val logger = LogManager.getLogger(Log4jSqlLogger::class)

        override fun log(context: StatementContext, transaction: Transaction) {
            logger.debug("SQL: ${context.expandArgs(transaction)}")
        }
    }

    init {
        Database.connect("jdbc:postgresql://${Config["url"]}:${Config["port"]}/${Config["database"]}",
                         driver = "org.postgresql.Driver",
                user = Config["username"],
                password = Config["password"])

        transaction {
            addLogger(Log4jSqlLogger)

            if (Config["resetDatabase"].toBoolean()) {
                SchemaUtils.drop(Publications, Citations, Keywords, KeywordsPublications)
            }

            SchemaUtils.create(Publications, Citations, Keywords, KeywordsPublications)
        }
    }

    fun store(articles: List<PubmedArticle>) {
        val keywordsForArticle = articles.map { it.keywordList.map { kw -> it.pmid to kw } }.flatten()
        val keywordSet = keywordsForArticle.map { it.second }.toSet()
        val citationsForArticle = articles.map { it.citationList.map { cit -> it.pmid to cit } }.flatten()

        transaction {
            addLogger(Log4jSqlLogger)

            Publications.batchInsert(articles, ignore = true) { article ->
                this[Publications.pmid] = article.pmid
                this[Publications.year] = article.year
                this[Publications.title] = article.title
                this[Publications.abstract] = article.abstractText
            }

            Citations.batchInsert(citationsForArticle) { citation ->
                this[Citations.pmidCiting] = citation.first
                this[Citations.pmidCited] = citation.second
            }

            val keywordIds = Keywords.batchInsert(keywordSet, ignore = true) { keyword ->
                this[Keywords.keyword] = keyword
            }

            val keywordIdsMap = keywordSet.zip(keywordIds) { kw, rs ->
                kw to rs.getValue(Keywords.id) as Int
            }.toMap()
            val keywordIdsForArticle = keywordsForArticle.map { it.first to (keywordIdsMap[it.second] ?: 0) }

            KeywordsPublications.batchInsert(keywordIdsForArticle) { keywordId ->
                this[KeywordsPublications.pmid] = keywordId.first
                this[KeywordsPublications.keywordId] = keywordId.second
            }
        }
    }
}