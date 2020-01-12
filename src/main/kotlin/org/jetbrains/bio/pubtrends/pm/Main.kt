package org.jetbrains.bio.pubtrends.pm

import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import java.io.BufferedReader
import java.io.FileReader
import java.util.*
import kotlin.system.exitProcess

fun main(args: Array<String>) {

    val logger = LogManager.getLogger("Pubtrends")

    with(OptionParser()) {
        accepts("resetDatabase", "Reset Database")

        accepts("lastId", "LastID").withRequiredArg().ofType(Int::class.java).defaultsTo(0)

        // Help option
        acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()

        logger.info("Arguments: \n" + Arrays.toString(args))

        val options = parse(*args)
        if (options.has("help")) {
            printHelpOn(System.err)
            exitProcess(0)
        }

        // Load configuration file
        val (config, configPath, settingsRoot) = Config.load()
        logger.info("Config\n" + BufferedReader(FileReader(configPath.toFile())).use {
            it.readLines().joinToString("\n")
        })

        logger.info("Crawling...")

        var resetDatabase = options.has("resetDatabase")
        if (resetDatabase) {
            logger.warn("RESETTING DATABASE")
        }

        var retry = 1
        var waitTime: Long = 1
        var isUpdateRequired = true
        logger.info("Retrying downloading after any problems.")
        while (isUpdateRequired) {
            try {
                logger.info("Init database connection")
                val dbHandler = Neo4jDatabaseHandler(
                        config["neo4jurl"].toString(),
                        config["neo4jport"].toString().toInt(),
                        config["neo4jusername"].toString(),
                        config["neo4jpassword"].toString(),
                        resetDatabase
                )
                // Avoid resetting database during further retries
                if (dbHandler.resetSucceeded) {
                    resetDatabase = false
                }

                dbHandler.use {
                    logger.info("Init Pubmed processor")
                    val pubmedXMLParser =
                            PubmedXMLParser(
                                    dbHandler,
                                    config["pm_parser_limit"].toString().toInt(),
                                    config["pm_batch_size"].toString().toInt()
                            )

                    logger.info("Init crawler")
                    val crawlerTSV = settingsRoot.resolve("pubmed_last.tsv")
                    val statsTSV = settingsRoot.resolve("pubmed_stats.tsv")
                    val collectStats = config["pm_collect_stats"].toString().toBoolean()
                    val pubmedCrawler = PubmedCrawler(pubmedXMLParser, collectStats, statsTSV, crawlerTSV)

                    val lastIdCmd = if (options.has("lastId")) options.valueOf("lastId").toString().toInt() else null

                    isUpdateRequired = if (retry == 1) {
                        pubmedCrawler.update(lastIdCmd)
                    } else {
                        pubmedCrawler.update(null)
                    }
                    waitTime = 1
                }
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