package org.jetbrains.bio.pubtrends.db

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction
import java.io.Closeable

open class PMPostgresWriter(
        host: String,
        port: Int,
        database: String,
        user: String,
        password: String
) : AbstractDBWriter<PubmedArticle>, Closeable {
    companion object Log4jSqlLogger : SqlLogger {
        private val logger = LogManager.getLogger(Log4jSqlLogger::class)

        override fun log(context: StatementContext, transaction: Transaction) {
            logger.debug("SQL: ${context.expandArgs(transaction)}")
        }
    }

    init {

        Database.connect(
                url = "jdbc:postgresql://$host:$port/$database",
                driver = "org.postgresql.Driver",
                user = user,
                password = password
        )

        transaction {
            addLogger(Log4jSqlLogger)

            val customTypeExists = exec(
                    "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'publicationtype');"
            ) { rs ->
                rs.next() && (rs.getBoolean("exists"))
            }

            if (customTypeExists == false) {
                exec(
                        "CREATE TYPE PublicationType AS ENUM " +
                                "(${PublicationType.values().joinToString(", ") { "'$it'" }});"
                )
            }

            SchemaUtils.create(PMPublications, PMCitations)
        }
    }

    override fun reset() {
        transaction {
            addLogger(Log4jSqlLogger)
            SchemaUtils.drop(PMPublications, PMCitations)
        }
    }

    override fun store(articles: List<PubmedArticle>) {
        val citationsForArticle = articles.map { it.citationList.toSet().map { cit -> it.pmid to cit } }.flatten()

        transaction {
            addLogger(Log4jSqlLogger)

            PMPublications.batchInsertOnDuplicateKeyUpdate(
                    articles, PMPublications.pmid,
                    listOf(
                            PMPublications.date, PMPublications.title, PMPublications.abstract,
                            PMPublications.keywords, PMPublications.mesh, PMPublications.type,
                            PMPublications.doi, PMPublications.aux
                    )
            ) { batch, article ->
                batch[pmid] = article.pmid
                batch[date] = article.date
                batch[title] = article.title.take(PUBLICATION_MAX_TITLE_LENGTH)
                if (article.abstractText != "") {
                    batch[abstract] = article.abstractText
                }

                batch[keywords] = article.keywordList.joinToString(separator = ", ")
                batch[mesh] = article.meshHeadingList.joinToString(separator = ", ")

                batch[type] = article.type
                batch[doi] = article.doi
                batch[aux] = article.auxInfo
            }

            PMCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[PMCitations.pmidOut] = citation.first
                this[PMCitations.pmidIn] = citation.second
            }
        }
    }

    override fun delete(ids: List<String>) {
        val intIds = ids.map { it.toInt() }
        transaction {
            addLogger(Log4jSqlLogger)

            PMPublications.deleteWhere { PMPublications.pmid inList intIds }
            PMCitations.deleteWhere {
                (PMCitations.pmidOut inList intIds) or (PMCitations.pmidIn inList intIds)
            }
        }
    }

    /**
     * Dummy function in order to implement Closeable interface.
     * No actions are needed in fact, Exposed should manage the connection pool.
     */
    override fun close() {}
}