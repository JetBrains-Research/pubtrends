package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import org.jetbrains.exposed.sql.*
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
        LOG.info("Initializing DB connection")
        Database.connect(
            url = "jdbc:postgresql://$host:$port/$database",
            driver = "org.postgresql.Driver",
            user = username,
            password = password
        )
        LOG.info("Init transaction started")
        transaction {
            LOG.info("Check and create enum publication type")
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

            LOG.info("Create schemas and tables")
            SchemaUtils.create(PMPublications, PMCitations)

            LOG.info("Adding tsv vector to table")
            exec("ALTER TABLE PMPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;")

            LOG.info("Creating index pm_title_abstract_index")
            exec(
                "CREATE INDEX IF NOT EXISTS pm_title_abstract_index ON PMPublications using GIN (tsv);"
            )
            LOG.info("Creating materialized view matview_pmcitations")
            exec(
                """
                create materialized view if not exists matview_pmcitations as
                SELECT pmid_in as pmid, COUNT(*) AS count
                FROM PMCitations C
                GROUP BY pmid_in
                HAVING COUNT(*) >= 3; -- Ignore tail of 0,1,2 cited papers
                create index if not exists PMCitation_matview_index on matview_pmcitations using hash (pmid);
                """
            )
            LOG.info("Creating index pmpublications_pmid_year")
            exec(
                "CREATE INDEX IF NOT EXISTS pmpublications_pmid_year ON pmpublications (pmid, year);"
            )
            LOG.info("Creating index pm_doi_index")
            exec(
                "CREATE INDEX IF NOT EXISTS pm_doi_index ON PMPublications using hash (doi);"
            )
        }
        LOG.info("Init transaction finished")
    }

    override fun reset() {
        LOG.info("Reset transaction started")
        transaction {
            LOG.info("Drop materialized view with index")
            exec(
                """
                drop index if exists PMCitation_matview_index;
                drop materialized view if exists matview_pmcitations;
                """
            )
            LOG.info("Drop other indexes")
            exec(
                """
                DROP INDEX IF EXISTS pm_doi_index;
                DROP INDEX IF EXISTS pm_title_abstract_index;
                DROP INDEX IF EXISTS pmpublications_pmid_year;
                """
            )
            LOG.info("Drop tables")
            SchemaUtils.drop(PMPublications, PMCitations)
            changed = true
        }
        LOG.info("Reset transaction finished")
    }

    override fun store(articles: List<PubmedArticle>, isBaseline: Boolean) {
        LOG.info("Store ${articles.size} articles (baseline: $isBaseline)")
        val citationsForArticle = articles.flatMap { it.citations.toSet().map { cit -> it.pmid to cit } }
        store(articles, citationsForArticle, isBaseline)
    }

    internal fun store(
        articles: List<PubmedArticle>,
        citationsList: List<Pair<Int, Int>>,
        isBaseline: Boolean = false
    ) {
        LOG.info("Store transaction started")
        transaction {
            val articlesToStore = if (isBaseline) {
                // For baseline files, filter out articles that already exist in the database
                // BUT include those with missing references that now have citations to add
                val pmids = articles.map { it.pmid }
                val existingPmids = PMPublications.select { PMPublications.pmid inList pmids }
                    .map { it[PMPublications.pmid] }
                    .toSet()

                // Find PMIDs that exist but have no citations AND have references in the incoming data
                val pmidsWithoutCitations = if (existingPmids.isNotEmpty()) {
                    val articlesWithReferences = articles.filter { it.citations.isNotEmpty() }.map { it.pmid }.toSet()
                    PMPublications.leftJoin(PMCitations, { PMPublications.pmid }, { PMCitations.pmidOut })
                        .select {
                            (PMPublications.pmid inList existingPmids) and
                            (PMCitations.pmidOut.isNull())
                        }
                        .map { it[PMPublications.pmid] }
                        .toSet()
                        .filter { it in articlesWithReferences }
                        .toSet()
                } else {
                    emptySet()
                }

                val filtered = articles.filter { it.pmid !in existingPmids || it.pmid in pmidsWithoutCitations }
                LOG.info("Baseline: ${articles.size} articles, ${existingPmids.size} already exist, ${pmidsWithoutCitations.size} with missing references that have new citations, ${filtered.size} to process")
                filtered
            } else {
                articles
            }

            if (articlesToStore.isEmpty()) {
                LOG.info("No articles to store")
                return@transaction
            }

            // For updates: insert or update records
            LOG.info("Batch insert or update publications")
            PMPublications.batchInsertOnDuplicateKeyUpdate(
                articlesToStore, listOf(PMPublications.pmid),
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
            ) { conflict, article ->
                conflict[pmid] = article.pmid
                conflict[title] = article.title.take(PUBLICATION_MAX_TITLE_LENGTH)
                if (article.abstract.isNotEmpty()) {
                    conflict[abstract] = article.abstract
                }
                conflict[year] = article.year
                conflict[keywords] = article.keywords.joinToString(",")
                conflict[mesh] = article.mesh.joinToString(",")
                conflict[type] = article.type
                conflict[doi] = article.doi
                conflict[aux] = article.aux
            }

            LOG.info("Update TSV vector")
            val vals = articlesToStore.map { it.pmid }.joinToString(",") { "($it)" }
            exec(
                """
                    UPDATE PMPublications
                    set tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                    WHERE pmid IN (VALUES $vals);
                    """
            )

            // Filter citations to only include those for articles we're actually storing
            val storedPmids = articlesToStore.map { it.pmid }.toSet()
            val citationsToStore = citationsList.filter { it.first in storedPmids }
            LOG.info("Batch insert citations (${citationsToStore.size} citations)")
            PMCitations.batchInsert(citationsToStore, ignore = true) { citation ->
                this[PMCitations.pmidOut] = citation.first
                this[PMCitations.pmidIn] = citation.second
            }
            changed = true
        }
        LOG.info("Store transaction finished")
    }

    override fun delete(ids: List<String>) {
        LOG.info("Delete ${ids.size} articles")
        // Postgresql org.postgresql.core.v3.QueryExecutorImpl
        // supports no more than Short.MAX_VALUE number of params
        require(ids.size <= Short.MAX_VALUE) { "Too many ids to remove ${ids.size}" }
        val intIds = ids.map { it.toInt() }
        LOG.info("Delete transaction started")
        transaction {
            LOG.info("Delete publications")
            PMPublications.deleteWhere { PMPublications.pmid inList intIds }
            LOG.info("Delete citations out")
            PMCitations.deleteWhere { PMCitations.pmidOut inList intIds }
            LOG.info("Delete citations in")
            PMCitations.deleteWhere { PMCitations.pmidIn inList intIds }
            changed = true
        }
        LOG.info("Delete transaction finished")
    }

    override fun close() {
        /**
         * No actions to close db connection is required: Exposed should manage the connection pool.
         */
        if (changed) {
            LOG.info("Close transaction started")
            transaction {
                LOG.info("Update material view matview_pmcitations")
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
            LOG.info("Close transaction finished")
        }
    }
}