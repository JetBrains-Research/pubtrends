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

    fun store(article : PubmedArticle) {
        transaction {
            addLogger(Log4jSqlLogger)

            Publications.insert {
                it[pmid] = article.pmid
                it[year] = article.year
                it[title] = article.title
                it[abstract] = article.abstractText
            }

            val keywordIds = Keywords.batchInsert(article.keywordList, ignore = true) { keyword ->
                this[Keywords.keyword] = keyword
            }

            KeywordsPublications.batchInsert(keywordIds) { keywordId ->
                this[KeywordsPublications.pmid] = article.pmid
                this[KeywordsPublications.keywordId] = keywordId.getValue(Keywords.id) as Int
            }

            Citations.batchInsert(article.citationList) { citation ->
                this[Citations.pmidCiting] = article.pmid
                this[Citations.pmidCited] = citation
            }
        }
    }
}