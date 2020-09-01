package org.jetbrains.bio.pubtrends.db

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction

open class PubmedPostgresWriter(
        host: String,
        port: Int,
        database: String,
        username: String,
        password: String
) : AbstractDBWriter<PubmedArticle> {
    companion object Log4jSqlLogger : SqlLogger {
        private val logger = LogManager.getLogger(Log4jSqlLogger::class)

        override fun log(context: StatementContext, transaction: Transaction) {
            logger.debug("SQL: ${context.expandArgs(transaction)}")
        }
    }

    private var changed = false

    init {

        Database.connect(
                url = "jdbc:postgresql://$host:$port/$database",
                driver = "org.postgresql.Driver",
                user = username,
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
            exec("ALTER TABLE PMPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;")
            exec(
                    """
                    CREATE INDEX IF NOT EXISTS
                    pm_title_abstract_index ON PMPublications using GIN (tsv);
                    """
            )
            exec(
                    """
                    create materialized view if not exists matview_pmcitations as
                    SELECT pmid_in as pmid, COUNT(*) AS count
                    FROM PMCitations C
                    GROUP BY pmid_in
                    HAVING COUNT(*) >= 3; -- Ignore tail of 0,1,2 cited papers
                    create index if not exists PMCitation_matview_index on matview_pmcitations (pmid);
                    """
            )
        }
    }

    override fun reset() {
        transaction {
            addLogger(Log4jSqlLogger)
            exec(
                    """
                    drop index if exists PMCitation_matview_index;
                    drop materialized view if exists matview_pmcitations;
                    """
            )
            SchemaUtils.drop(PMPublications, PMCitations)
            exec("DROP INDEX IF EXISTS pm_title_abstract_index;")
            changed = true
        }
    }

    override fun store(articles: List<PubmedArticle>) {
        val citationsForArticle = articles.map { it.citations.toSet().map { cit -> it.pmid to cit } }.flatten()

        transaction {
            addLogger(Log4jSqlLogger)

            PMPublications.batchInsertOnDuplicateKeyUpdate(
                    articles, PMPublications.pmid,
                    listOf(
                            PMPublications.date,
                            PMPublications.title,
                            PMPublications.abstract,
                            PMPublications.keywords,
                            PMPublications.mesh,
                            PMPublications.type,
                            PMPublications.doi,
                            PMPublications.aux
                    )
            ) { batch, article ->
                batch[pmid] = article.pmid
                batch[date] = article.date
                batch[title] = article.title.take(PUBLICATION_MAX_TITLE_LENGTH)
                if (article.abstract != "") {
                    batch[abstract] = article.abstract
                }
                batch[keywords] = article.keywords.joinToString(",")
                batch[mesh] = article.mesh.joinToString(",")
                batch[type] = article.type
                batch[doi] = article.doi
                batch[aux] = article.aux
            }

            PMCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[PMCitations.pmidOut] = citation.first
                this[PMCitations.pmidIn] = citation.second
            }

            // Update TSV vector
            val vals = articles.map { it.pmid }.joinToString(",") { "($it)" }
            exec(
                    """
                    UPDATE PMPublications
                    set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') || 
                                setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                    WHERE pmid IN (VALUES $vals);
                    """
            )
            changed = true
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
            changed = true
        }
    }

    override fun close() {
        /**
         * No actions to close db connection is required: Exposed should manage the connection pool.
         */
        if (changed) {
            transaction {
                addLogger(Log4jSqlLogger)
                exec("""
                    do
                    ${"$$"}
                    begin
                    IF (select matviewname from pg_matviews where matviewname = 'matview_pmcitations') THEN
                        refresh materialized view matview_pmcitations;
                    END IF;
                    end;
                    ${"$$"}
                    """)
            }
        }
    }
}