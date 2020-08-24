package org.jetbrains.bio.pubtrends.db

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.ss.SemanticScholarArticle
import org.jetbrains.bio.pubtrends.ss.crc32id
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction
import java.io.Closeable

open class SSPostgresWriter(
        host: String,
        port: Int,
        database: String,
        user: String,
        password: String
)
    : AbstractDBWriter<SemanticScholarArticle>, Closeable {
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
            SchemaUtils.create(SSPublications, SSCitations)
            exec("ALTER TABLE SSPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;")
            exec(
                    "CREATE INDEX IF NOT EXISTS " +
                            "ss_title_abstract_index ON SSPublications using GIN (tsv);"
            )

        }
    }

    override fun reset() {
        transaction {
            addLogger(Log4jSqlLogger)
            SchemaUtils.drop(SSPublications, SSCitations)
            exec("DROP INDEX IF EXISTS ss_title_abstract_index;")
        }
    }

    override fun store(articles: List<SemanticScholarArticle>) {
        val citationsList = articles.map { it.citationList.distinct().map { cit -> it.ssid to cit } }.flatten()

        transaction {
            SSPublications.batchInsert(articles, ignore = true) { article ->
                this[SSPublications.ssid] = article.ssid
                this[SSPublications.crc32id] = crc32id(article.ssid)
                this[SSPublications.pmid] = article.pmid
                this[SSPublications.abstract] = article.abstract
                this[SSPublications.keywords] = article.keywords
                this[SSPublications.title] = article.title
                this[SSPublications.year] = article.year
                this[SSPublications.doi] = article.doi
                this[SSPublications.aux] = article.aux
            }

            SSCitations.batchInsert(citationsList, ignore = true) { citation ->
                this[SSCitations.id_out] = citation.first
                this[SSCitations.id_in] = citation.second
                this[SSCitations.crc32id_out] = crc32id(citation.first)
                this[SSCitations.crc32id_in] = crc32id(citation.second)
            }
            // Update TSV vector
            val vals = articles.map { it.ssid }.joinToString(",") { "('$it')" }
            exec(
                    "UPDATE SSPublications\n" +
                            "set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') || \n" +
                            "   setweight(to_tsvector('english', coalesce(abstract, '')), 'B')\n" +
                            "WHERE ssid IN (VALUES $vals);"
            )

        }
    }


    override fun delete(ids: List<String>) {
        throw IllegalStateException("delete is not supported")
    }

    /**
     * Dummy function in order to implement Closeable interface.
     * No actions are needed in fact, Exposed should manage the connection pool.
     */
    override fun close() {}
}

