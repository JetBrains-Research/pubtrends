package org.jetbrains.bio.pubtrends.pm

import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.db.AbstractDBWriter
import org.jetbrains.bio.pubtrends.db.PubmedNeo4JWriter
import org.jetbrains.bio.pubtrends.db.PubmedPostgresWriter
import java.nio.file.Files
import kotlin.system.exitProcess

object PubmedLoader {
    @JvmStatic
    fun main(args: Array<String>) {

        val logger = LogManager.getLogger("Pubtrends")

        with(OptionParser()) {
            accepts("resetDatabase", "Reset Database")
            accepts("fillDatabase", "Create and fill database with articles")

            accepts("lastId", "LastID").withRequiredArg().ofType(Int::class.java).defaultsTo(0)

            // Help option
            acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()

            logger.info("Arguments: \n" + args.contentToString())

            val options = parse(*args)
            if (options.has("help")) {
                printHelpOn(System.err)
                exitProcess(0)
            }

            // Load configuration file
            val (config, configPath, settingsRoot) = Config.load()
            logger.info("Config path: $configPath")

            val dbWriter: AbstractDBWriter<PubmedArticle>
            if (!(config["neo4j_host"]?.toString()).isNullOrBlank()) {
                logger.info("Init Neo4j database connection")
                dbWriter = PubmedNeo4JWriter(
                        config["neo4j_host"]!!.toString(),
                        config["neo4j_port"]!!.toString().toInt(),
                        config["neo4j_username"]!!.toString(),
                        config["neo4j_password"]!!.toString()
                )
            } else if (!(config["postgres_host"]?.toString()).isNullOrBlank()) {
                logger.info("Init Postgresql database connection")
                dbWriter = PubmedPostgresWriter(
                        config["postgres_host"]!!.toString(),
                        config["postgres_port"]!!.toString().toInt(),
                        config["postgres_database"]!!.toString(),
                        config["postgres_username"]!!.toString(),
                        config["postgres_password"]!!.toString()
                )
            } else {
                throw IllegalStateException("No database configured")
            }

            val writerName = dbWriter.javaClass.simpleName.toLowerCase()
            val pubmedLastIdFile = settingsRoot.resolve("${writerName}_last.tsv")
            val pubmedStatsFile = settingsRoot.resolve("${writerName}_stats.tsv")

            dbWriter.use {
                if (options.has("resetDatabase")) {
                    logger.info("Resetting database")
                    dbWriter.reset()
                    Files.deleteIfExists(pubmedLastIdFile)
                    Files.deleteIfExists(pubmedStatsFile)
                }

                if (options.has("fillDatabase")) {
                    logger.info("Checking Pubmed FTP...")
                    var retry = 1
                    var waitTime: Long = 1
                    var isUpdateRequired = true
                    logger.info("Retrying downloading after any problems.")
                    while (isUpdateRequired) {
                        try {
                            logger.info("Init Pubmed processor")
                            val pubmedXMLParser =
                                    PubmedXMLParser(dbWriter, config["loader_batch_size"].toString().toInt())

                            logger.info("Init crawler")
                            val collectStats = config["loader_collect_stats"].toString().toBoolean()
                            val pubmedCrawler = PubmedCrawler(pubmedXMLParser, collectStats,
                                    pubmedStatsFile, pubmedLastIdFile)

                            val lastIdCmd = if (options.has("lastId"))
                                options.valueOf("lastId").toString().toInt()
                            else
                                null

                            isUpdateRequired = if (retry == 1) {
                                pubmedCrawler.update(lastIdCmd)
                            } else {
                                pubmedCrawler.update(null)
                            }
                            waitTime = 1
                        } catch (e: PubmedCrawlerException) {
                            logger.error(e)
                            isUpdateRequired = true

                            logger.info("Waiting for $waitTime seconds...")
                            Thread.sleep(waitTime * 1000)
                            logger.info("Retry #$retry")
                            retry += 1

                            if (waitTime < 1024) {
                                waitTime *= 2
                            }
                        }
                    }
                    logger.info("Done crawling.")
                }
            }
        }
    }
}
