package org.jetbrains.bio.pubtrends

import org.jetbrains.bio.pubtrends.db.PubmedPostgresWriter
import org.jetbrains.bio.pubtrends.db.SemanticScholarPostgresWriter
import org.jetbrains.bio.pubtrends.pm.*
import org.jetbrains.bio.pubtrends.ss.SS_EXTRA_ARTICLES
import org.jetbrains.bio.pubtrends.ss.SS_EXTRA_CITATIONS
import org.jetbrains.bio.pubtrends.ss.SS_REQUIRED_ARTICLES
import org.jetbrains.bio.pubtrends.ss.SS_REQUIRED_CITATIONS
import org.slf4j.LoggerFactory

/**
 * Class to save single publication into test db.
 *
 * Usage: DBWriter <WriterClass> --host <host> --port <port> --database <db> --username <user> --password <pass>
 */
class DBWriter {
    companion object {
        private val LOG = LoggerFactory.getLogger(DBWriter::class.java)

        data class DbParams(
            val host: String,
            val port: Int,
            val database: String,
            val username: String,
            val password: String
        )

        private fun parseArgs(args: Array<String>): Pair<List<String>, DbParams> {
            val writers = mutableListOf<String>()
            var host = "localhost"
            var port = 5432
            var database = "test_pubtrends"
            var username = "biolabs"
            var password = "mysecretpassword"

            var i = 0
            while (i < args.size) {
                when (args[i]) {
                    "--host" -> { host = args[++i] }
                    "--port" -> { port = args[++i].toInt() }
                    "--database" -> { database = args[++i] }
                    "--username" -> { username = args[++i] }
                    "--password" -> { password = args[++i] }
                    else -> writers.add(args[i])
                }
                i++
            }
            return Pair(writers, DbParams(host, port, database, username, password))
        }

        @JvmStatic
        fun main(args: Array<String>) {
            val (writers, db) = parseArgs(args)

            for (writer in writers) {
                LOG.info("Processing $writer")
                when (writer) {
                    PubmedPostgresWriter::class.java.simpleName -> {
                        createPubmedPostgresWriter(db).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createPubmedPostgresWriter(db).use {
                            LOG.info("Store")
                            it.store(
                                PM_REQUIRED_ARTICLES + PM_EXTRA_ARTICLES + listOf(PM_EXTRA_ARTICLE),
                                PM_INNER_CITATIONS + PM_OUTER_CITATIONS
                            )
                        }
                    }

                    SemanticScholarPostgresWriter::class.java.simpleName -> {
                        createSemanticScholarPostgresWriter(
                            db,
                            initIndexesAndMatView = false,
                            finishFillDatabase = false
                        ).use {
                            LOG.info("Reset")
                            it.reset()
                        }
                        createSemanticScholarPostgresWriter(
                            db,
                            initIndexesAndMatView = false,
                            finishFillDatabase = false
                        ).use {
                            LOG.info("Store")
                            it.store(
                                SS_REQUIRED_ARTICLES + SS_EXTRA_ARTICLES,
                                SS_REQUIRED_CITATIONS + SS_EXTRA_CITATIONS
                            )
                        }
                        // Create indexes and update them
                        createSemanticScholarPostgresWriter(
                            db,
                            initIndexesAndMatView = true,
                            finishFillDatabase = true
                        ).close()
                    }

                    else -> LOG.error("Unknown arg $writer")
                }
                LOG.info("Done")
            }
            if (writers.isEmpty())
                println("Welcome to Pubtrends! See README.md for deployment instructions.")
        }

        private fun createSemanticScholarPostgresWriter(
            db: DbParams,
            initIndexesAndMatView: Boolean,
            finishFillDatabase: Boolean
        ): SemanticScholarPostgresWriter {
            return SemanticScholarPostgresWriter(
                db.host, db.port, db.database, db.username, db.password,
                initIndexesAndMatView, finishFillDatabase
            )
        }

        private fun createPubmedPostgresWriter(db: DbParams): PubmedPostgresWriter {
            return PubmedPostgresWriter(
                db.host, db.port, db.database, db.username, db.password
            )
        }
    }
}