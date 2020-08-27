package org.jetbrains.bio.pubtrends

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.db.PubmedNeo4JWriter
import org.jetbrains.bio.pubtrends.db.PubmedPostgresWriter
import org.jetbrains.bio.pubtrends.db.SemanticScholarNeo4JWriter
import org.jetbrains.bio.pubtrends.db.SemanticScholarPostgresWriter
import org.jetbrains.bio.pubtrends.pm.PublicationType
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import org.jetbrains.bio.pubtrends.ss.*
import org.joda.time.DateTime
import org.joda.time.DateTimeZone
import java.util.*

/**
 * Class to save single publication into test db
 */
class DBWriter {
    companion object {
        private val LOG = LogManager.getLogger("DBWriter")

        private val PUBMED_ARTICLE = PubmedArticle(
                pmid = 1,
                date = DateTime(1986, 4, 7, 12, 0, DateTimeZone.UTC),
                title = "Test title 1",
                abstract = "Test abstract 2",
                aux = org.jetbrains.bio.pubtrends.pm.Aux(
                        authors = listOf(org.jetbrains.bio.pubtrends.pm.Author("Genius1")),
                        journal = org.jetbrains.bio.pubtrends.pm.Journal("Pravda")
                ),
                keywords = listOf("Keyword1", "Keyword2"),
                mesh = listOf("Term1", "Term2", "Term3"),
                type = PublicationType.TechnicalReport
        )

        private val SEMANTIC_SCHOLAR_ARTICLE = SemanticScholarArticle(
                ssid = "03029e4427cfe66c3da6257979dc2d5b6eb3a0e4",
                pmid = 2252909,
                title = "Test title 1",
                abstract = "Test abstract 2",
                year = 2020,
                doi = "10.1101/2020.05.10.087023",
                aux = Aux(journal = Journal(name = "Nature Aging", volume = "1", pages = "1-6"),
                        authors = listOf(Author(name = "Genius")),
                        venue = "Nature",
                        links = Links(pdfUrls = listOf("https://doi.org/10.1101/2020.05.10.087023"))
                ))

        @JvmStatic
        fun main(args: Array<String>) {
            // Load configuration file
            val (config, _, _) = Config.load()

            for (arg in args) {
                LOG.info("Processing $arg")
                when (arg) {
                    PubmedNeo4JWriter::class.java.simpleName -> {
                        createPubmedNeo4JWriter(config).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createPubmedNeo4JWriter(config).use {
                            LOG.info("Store")
                            it.store(listOf(PUBMED_ARTICLE))
                        }
                    }
                    PubmedPostgresWriter::class.java.simpleName -> {
                        createPubmedPostgresWriter(config).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createPubmedPostgresWriter(config).use {
                            LOG.info("Store")
                            it.store(listOf(PUBMED_ARTICLE))
                        }
                    }
                    SemanticScholarNeo4JWriter::class.java.simpleName -> {
                        createSemanticScholarNeo4JWriter(config).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createSemanticScholarNeo4JWriter(config).use {
                            LOG.info("Store")
                            it.store(listOf(SEMANTIC_SCHOLAR_ARTICLE))
                        }
                    }
                    SemanticScholarPostgresWriter::class.java.simpleName -> {
                        createSemanticScholarPostgresWriter(config).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createSemanticScholarPostgresWriter(config).use {
                            LOG.info("Store")
                            it.store(listOf(SEMANTIC_SCHOLAR_ARTICLE))
                        }
                    }
                    else -> LOG.error("Unknown arg $arg")
                }
                LOG.info("Done")
            }
            if (args.isEmpty())
                println("Welcome to Pubtrends! See README.md for deployment instructions.")
        }

        private fun createSemanticScholarPostgresWriter(config: Properties): SemanticScholarPostgresWriter {
            return SemanticScholarPostgresWriter(
                    config["test_postgres_host"]!!.toString(),
                    config["test_postgres_port"]!!.toString().toInt(),
                    config["test_postgres_database"]!!.toString(),
                    config["test_postgres_username"]!!.toString(),
                    config["test_postgres_password"]!!.toString()
            )
        }

        private fun createSemanticScholarNeo4JWriter(config: Properties): SemanticScholarNeo4JWriter {
            return SemanticScholarNeo4JWriter(
                    config["test_neo4j_host"]!!.toString(),
                    config["test_neo4j_port"]!!.toString().toInt(),
                    config["test_neo4j_username"]!!.toString(),
                    config["test_neo4j_password"]!!.toString()
            )
        }

        private fun createPubmedNeo4JWriter(config: Properties): PubmedNeo4JWriter {
            return PubmedNeo4JWriter(
                    config["test_neo4j_host"]!!.toString(),
                    config["test_neo4j_port"]!!.toString().toInt(),
                    config["test_neo4j_username"]!!.toString(),
                    config["test_neo4j_password"]!!.toString()
            )
        }

        private fun createPubmedPostgresWriter(config: Properties): PubmedPostgresWriter {
            return PubmedPostgresWriter(
                    config["test_postgres_host"]!!.toString(),
                    config["test_postgres_port"]!!.toString().toInt(),
                    config["test_postgres_database"]!!.toString(),
                    config["test_postgres_username"]!!.toString(),
                    config["test_postgres_password"]!!.toString()
            )
        }
    }
}