package org.jetbrains.bio.pubtrends.db

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.ss.SemanticScholarArticle
import org.jetbrains.bio.pubtrends.ss.crc32id
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction

open class SemanticScholarPostgresWriter(
    host: String,
    port: Int,
    database: String,
    username: String,
    password: String,
    private val finishFillDatabase: Boolean
) : AbstractDBWriter<SemanticScholarArticle> {
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
            user = username,
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
            exec(
                """
                    create materialized view if not exists matview_sscitations as
                    SELECT ssid_in as ssid, crc32id_in as crc32id, COUNT(*) AS count
                    FROM SSCitations C
                    GROUP BY ssid, crc32id
                    HAVING COUNT(*) >= 3; -- Ignore tail of 0,1,2 cited papers
                    create index if not exists SSCitation_matview_index on matview_sscitations (crc32id);
                    """
            )
            exec(
                """
                    CREATE INDEX IF NOT EXISTS
                    sspublications_ssid_year ON sspublications (ssid, year);
                    """
            )
        }
    }

    override fun reset() {
        transaction {
            addLogger(Log4jSqlLogger)
            exec(
                """
                    drop index if exists SSCitation_matview_index;
                    drop materialized view if exists matview_sscitations;
                    """
            )
            SchemaUtils.drop(SSPublications, SSCitations)
            exec("DROP INDEX IF EXISTS ss_title_abstract_index;")
        }
    }

    override fun store(articles: List<SemanticScholarArticle>) {
        val citationsList = articles.map { it.citations.distinct().map { cit -> it.ssid to cit } }.flatten()

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
                this[SSCitations.ssid_out] = citation.first
                this[SSCitations.ssid_in] = citation.second
                this[SSCitations.crc32id_out] = crc32id(citation.first)
                this[SSCitations.crc32id_in] = crc32id(citation.second)
            }
            // Update TSV vector
            val vals = articles.map { it.ssid }.joinToString(",") { "('$it', ${crc32id(it)})" }
            exec(
                "UPDATE SSPublications\n" +
                        "set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') || \n" +
                        "   setweight(to_tsvector('english', coalesce(abstract, '')), 'B')\n" +
                        "WHERE (ssid, crc32id) IN (VALUES $vals);"
            )
        }
    }


    override fun delete(ids: List<String>) {
        throw IllegalStateException("delete is not supported")
    }

    override fun close() {
        if (finishFillDatabase) {
            logger.info("Refreshing matview_sscitations")
            transaction {
                addLogger(PubmedPostgresWriter)
                exec(
                    """
                    do
                    $$
                    begin
                    IF exists (select matviewname from pg_matviews where matviewname = 'matview_sscitations') THEN
                        refresh materialized view matview_sscitations;
                    END IF;
                    end;
                    $$;
                    """
                )
            }
            logger.info("Done refreshing matview_sscitations")
        }
    }
}

