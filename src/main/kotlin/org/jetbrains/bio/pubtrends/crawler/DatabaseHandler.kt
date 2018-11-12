package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction

class DatabaseHandler(username : String, password : String, reset : Boolean = false) {
    companion object Log4jSqlLogger : SqlLogger {
        private val logger = LogManager.getLogger(Log4jSqlLogger::class)

        override fun log(context: StatementContext, transaction: Transaction) {
            logger.debug("SQL: ${context.expandArgs(transaction)}")
        }
    }

    init {
        Database.connect("jdbc:postgresql://localhost:5432/pubmed",
                         driver = "org.postgresql.Driver",
                         user = username,
                         password = password)

        transaction {
            addLogger(Log4jSqlLogger)

            if (reset) {
                clear()
            }

            SchemaUtils.create(Publications, Citations, Keywords, KeywordsPublications)
        }
    }

    val lastModification : Long = transaction {
        exec("SELECT pg_xact_commit_timestamp(xmin) AS timestamp " +
                "FROM Publications " +
                "ORDER BY timestamp DESC " +
                "LIMIT 1;") {
            if (it.next()) it.getTimestamp("timestamp").time else 0
        } ?: 0
    }

    private fun clear() = transaction {
        SchemaUtils.drop(Publications, Citations, Keywords, KeywordsPublications)
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