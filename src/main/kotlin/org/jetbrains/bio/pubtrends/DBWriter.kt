package org.jetbrains.bio.pubtrends

import org.jetbrains.bio.pubtrends.db.PubmedPostgresWriter
import org.jetbrains.bio.pubtrends.db.SemanticScholarPostgresWriter
import org.jetbrains.bio.pubtrends.pm.*
import org.jetbrains.bio.pubtrends.ss.*
import org.slf4j.LoggerFactory
import java.util.*

/**
 * Class to save single publication into test db
 */
class DBWriter {
    companion object {
        private val LOG = LoggerFactory.getLogger(DBWriter::class.java)



        @JvmStatic
        fun main(args: Array<String>) {
            // Load configuration file
            val (config, _, _) = Config.load()

            for (arg in args) {
                LOG.info("Processing $arg")
                when (arg) {
                    PubmedPostgresWriter::class.java.simpleName -> {
                        createPubmedPostgresWriter(config).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createPubmedPostgresWriter(config).use {
                            LOG.info("Store")
                            it.store(
                                PM_REQUIRED_ARTICLES + PM_EXTRA_ARTICLES + listOf(PM_EXTRA_ARTICLE),
                                PM_INNER_CITATIONS + PM_OUTER_CITATIONS
                            )
                        }
                    }

                    SemanticScholarPostgresWriter::class.java.simpleName -> {
                        createSemanticScholarPostgresWriter(config).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createSemanticScholarPostgresWriter(config).use {
                            LOG.info("Store")
                            it.store(
                                SS_REQUIRED_ARTICLES + SS_EXTRA_ARTICLES,
                                SS_REQUIRED_CITATIONS + SS_EXTRA_CITATIONS
                            )
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
                config["test_postgres_password"]!!.toString(),
                true,
                true
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