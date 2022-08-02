package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.transaction
import org.slf4j.LoggerFactory

open class PubmedPostgresWriter(
    host: String,
    port: Int,
    database: String,
    username: String,
    password: String
) : AbstractDBWriter<PubmedArticle> {

    companion object {
        private val LOG = LoggerFactory.getLogger(PubmedPostgresWriter::class.java)
    }

    private var changed = false

    init {
        LOG.debug("Initializing DB connection")
        Database.connect(
            url = "jdbc:postgresql://$host:$port/$database",
            driver = "org.postgresql.Driver",
            user = username,
            password = password
        )
        LOG.debug("Init transaction started")
        transaction {
            LOG.debug("Check and create enum publication type")
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

            LOG.debug("Create schemas and tables")
            SchemaUtils.create(PMPublications, PMCitations)

            LOG.debug("Adding tsv vector to table")
            exec("ALTER TABLE PMPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;")

            LOG.debug("Creating index pm_title_abstract_index")
            exec(
                "CREATE INDEX IF NOT EXISTS pm_title_abstract_index ON PMPublications using GIN (tsv);"
            )
            LOG.debug("Creating materialized view matview_pmcitations")
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
            LOG.debug("Creating index pmpublications_pmid_year")
            exec(
                "CREATE INDEX IF NOT EXISTS pmpublications_pmid_year ON pmpublications (pmid, year);"
            )
            LOG.debug("Creating index pm_doi_index")
            exec(
                "CREATE INDEX IF NOT EXISTS pm_doi_index ON PMPublications using HASH (doi);"
            )
        }
        LOG.debug("Init transaction finished")
    }

    override fun reset() {
        LOG.debug("Reset transaction started")
        transaction {
            LOG.debug("Drop materialized view with index")
            exec(
                """
                drop index if exists PMCitation_matview_index;
                drop materialized view if exists matview_pmcitations;
                """
            )
            LOG.debug("Drop other indexes")
            exec(
                """
                DROP INDEX IF EXISTS pm_doi_index;
                DROP INDEX IF EXISTS pm_title_abstract_index;
                DROP INDEX IF EXISTS pmpublications_pmid_year;
                """
            )
            LOG.debug("Drop tables")
            SchemaUtils.drop(PMPublications, PMCitations)
            changed = true
        }
        LOG.debug("Reset transaction finished")
    }

    override fun store(articles: List<PubmedArticle>) {
        LOG.debug("Store ${articles.size} articles")
        val citationsForArticle = articles.map { it.citations.toSet().map { cit -> it.pmid to cit } }.flatten()
        LOG.debug("Store transaction started")
        transaction {
            LOG.debug("Batch insert or update publications")
            PMPublications.batchInsertOnDuplicateKeyUpdate(
                articles, PMPublications.pmid,
                listOf(
                    PMPublications.title,
                    PMPublications.abstract,
                    PMPublications.year,
                    PMPublications.keywords,
                    PMPublications.mesh,
                    PMPublications.type,
                    PMPublications.doi,
                    PMPublications.aux
                )
            ) { batch, article ->
                batch[pmid] = article.pmid
                batch[title] = article.title.take(PUBLICATION_MAX_TITLE_LENGTH)
                if (article.abstract != "") {
                    batch[abstract] = article.abstract
                }
                batch[year] = article.year
                batch[keywords] = article.keywords.joinToString(",")
                batch[mesh] = article.mesh.joinToString(",")
                batch[type] = article.type
                batch[doi] = article.doi
                batch[aux] = article.aux
            }
            LOG.debug("Update TSV vector")
            val vals = articles.map { it.pmid }.joinToString(",") { "($it)" }
            exec(
                """
                UPDATE PMPublications
                set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') || 
                    setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                WHERE pmid IN (VALUES $vals);
                """
            )
            LOG.debug("Batch insert citations")
            PMCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[PMCitations.pmidOut] = citation.first
                this[PMCitations.pmidIn] = citation.second
            }
            changed = true
        }
        LOG.debug("Store transaction finished")
    }

    override fun delete(ids: List<String>) {
        LOG.debug("Delete ${ids.size} articles")
        // Postgresql org.postgresql.core.v3.QueryExecutorImpl
        // supports no more than Short.MAX_VALUE number of params
        require(ids.size <= Short.MAX_VALUE) { "Too many ids to remove ${ids.size}" }
        val intIds = ids.map { it.toInt() }
        LOG.debug("Delete transaction started")
        transaction {
            LOG.debug("Delete publications")
            PMPublications.deleteWhere { PMPublications.pmid inList intIds }
            LOG.debug("Delete citations out")
            PMCitations.deleteWhere { PMCitations.pmidOut inList intIds }
            LOG.debug("Delete citations in")
            PMCitations.deleteWhere { PMCitations.pmidIn inList intIds }
            changed = true
        }
        LOG.debug("Delete transaction finished")
    }

    override fun close() {
        /**
         * No actions to close db connection is required: Exposed should manage the connection pool.
         */
        if (changed) {
            LOG.debug("Close transaction started")
            transaction {
                LOG.debug("Update material view matview_pmcitations")
                exec(
                    """
                    do
                    ${"$$"}
                    begin
                    IF exists (select matviewname from pg_matviews where matviewname = 'matview_pmcitations') THEN
                        refresh materialized view matview_pmcitations;
                    END IF;
                    end;
                    ${"$$"};
                    """
                )
            }
            LOG.debug("Close transaction finished")
        }
    }
}