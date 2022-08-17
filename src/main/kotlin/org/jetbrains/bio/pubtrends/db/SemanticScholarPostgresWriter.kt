package org.jetbrains.bio.pubtrends.db

import org.jetbrains.bio.pubtrends.ss.SemanticScholarArticle
import org.jetbrains.bio.pubtrends.ss.crc32id
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.batchInsert
import org.jetbrains.exposed.sql.transactions.transaction
import org.slf4j.LoggerFactory

open class SemanticScholarPostgresWriter(
    host: String,
    port: Int,
    database: String,
    username: String,
    password: String,
    private val finishFillDatabase: Boolean
) : AbstractDBWriter<SemanticScholarArticle> {
    companion object {
        private val LOG = LoggerFactory.getLogger(SemanticScholarPostgresWriter::class.java)
    }

    init {
        LOG.info("Initializing DB connection")
        Database.connect(
            url = "jdbc:postgresql://$host:$port/$database",
            driver = "org.postgresql.Driver",
            user = username,
            password = password
        )
        LOG.info("Init transaction starting")
        transaction {
            LOG.info("Creating schema")
            SchemaUtils.create(SSPublications, SSCitations)
            LOG.info("Adding TSV column")
            exec("ALTER TABLE SSPublications ADD COLUMN IF NOT EXISTS tsv TSVECTOR;")
            LOG.info("Creating index ss_title_abstract_index")
            exec(
                "CREATE INDEX IF NOT EXISTS ss_title_abstract_index ON SSPublications using GIN (tsv);"
            )
            LOG.info("Creating citations material view matview_sscitations")
            exec(
                """
                create materialized view if not exists matview_sscitations as
                SELECT ssid_in as ssid, crc32id_in as crc32id, COUNT(*) AS count
                FROM SSCitations C
                GROUP BY ssid, crc32id
                HAVING COUNT(*) >= 3; -- Ignore tail of 0,1,2 cited papers
                create index if not exists SSCitation_matview_index on matview_sscitations using hash(crc32id);
                """
            )
            LOG.info("Creating index sspublications_ssid_year")
            exec(
                """
                CREATE INDEX IF NOT EXISTS
                sspublications_ssid_year ON sspublications (ssid, year);
                """
            )
            LOG.info("Creating index sspublications_doi_index")
            exec(
                "create index if not exists sspublications_doi_index on sspublications using hash(doi);"
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
                drop index if exists SSCitation_matview_index;
                drop materialized view if exists matview_sscitations;
                """
            )
            LOG.info("Drop other indexes")
            exec(
                """
                drop index if exists sspublications_ssid_year;
                drop index if exists sspublications_doi_index;
                DROP INDEX IF EXISTS ss_title_abstract_index;
                """
            )
            LOG.info("Drop tables")
            SchemaUtils.drop(SSPublications, SSCitations)
        }
        LOG.info("Reset transaction finished")
    }

    override fun store(articles: List<SemanticScholarArticle>) {
        LOG.info("Store batch of ${articles.size} articles")
        val citationsList = articles.map { it.citations.distinct().map { cit -> it.ssid to cit } }.flatten()
        store(articles, citationsList)
    }

    internal fun store(
        articles: List<SemanticScholarArticle>,
        citationsList: List<Pair<String, String>>
    ) {
        LOG.info("Store batch transaction started")
        transaction {
            LOG.info("Batch insert articles")
            SSPublications.batchInsertOnDuplicateKeyUpdate(
                articles, listOf(SSPublications.crc32id, SSPublications.ssid),
                listOf(
                    SSPublications.pmid,
                    SSPublications.title,
                    SSPublications.abstract,
                    SSPublications.keywords,
                    SSPublications.year,
                    SSPublications.doi,
                    SSPublications.aux
                )
            ) { conflict, article ->
                conflict[ssid] = article.ssid
                conflict[crc32id] = crc32id(article.ssid)
                conflict[pmid] = article.pmid
                conflict[title] = article.title.take(PUBLICATION_MAX_TITLE_LENGTH)
                if (!article.abstract.isNullOrEmpty()) {
                    conflict[abstract] = article.abstract
                }
                conflict[keywords] = article.keywords
                conflict[year] = article.year
                conflict[doi] = article.doi
                conflict[aux] = article.aux
            }
            LOG.info("Update TSV vector")
            val vals = articles.joinToString(",") { "('${it.ssid}', ${it.crc32id})" }
            exec(
                """
                    UPDATE SSPublications
                    SET tsv = setweight(to_tsvector('english', coalesce(title, '')), 'A') || 
                        setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                    WHERE (ssid, crc32id) IN (VALUES $vals);
                    """
            )
            LOG.info("Batch insert citations list")
            SSCitations.batchInsert(citationsList, ignore = true) { citation ->
                this[SSCitations.ssid_out] = citation.first
                this[SSCitations.ssid_in] = citation.second
                this[SSCitations.crc32id_out] = crc32id(citation.first)
                this[SSCitations.crc32id_in] = crc32id(citation.second)
            }
        }
        LOG.info("Store batch transaction finished")
    }


    override fun delete(ids: List<String>) {
        throw IllegalStateException("delete is not supported")
    }

    override fun close() {
        if (finishFillDatabase) {
            LOG.info("Close transaction started")
            transaction {
                LOG.info("Refreshing materialized view")
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
            LOG.info("Close transaction finished")
        }
    }
}

