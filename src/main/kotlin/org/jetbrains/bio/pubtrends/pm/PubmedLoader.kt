package org.jetbrains.bio.pubtrends.pm

import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import java.io.BufferedReader
import java.io.FileReader
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

            logger.info("Init Neo4j database connection")
            val dbHandler = PMNeo4jDatabaseHandler(
                    config["neo4jhost"].toString(),
                    config["neo4jport"].toString().toInt(),
                    config["neo4jusername"].toString(),
                    config["neo4jpassword"].toString()
            )
            val pubmedLastIdFile = settingsRoot.resolve("pubmed_last.tsv")
            val pubmedStatsFile = settingsRoot.resolve("pubmed_stats.tsv")

            dbHandler.use {
                if (options.has("resetDatabase")) {
                    logger.info("Resetting database")
                    dbHandler.resetDatabase()
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
                                    PubmedXMLParser(dbHandler, config["loader_batch_size"].toString().toInt())

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
