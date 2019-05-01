package org.jetbrains.bio.pubtrends

import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.crawler.PostgresqlDatabaseHandler
import org.jetbrains.bio.pubtrends.crawler.PubmedCrawler
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler
import java.io.BufferedReader
import java.io.FileReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*

fun main(args: Array<String>) {

    val logger = LogManager.getLogger("Pubtrends")

    with(OptionParser()) {
        accepts("resetDatabase", "Reset Database")

        accepts("retry", "Keep retrying downloading new files after any problems")
        accepts("lastCheck", "Last check").withRequiredArg().ofType(Int::class.java).defaultsTo(0)
        accepts("lastId", "LastID").withRequiredArg().ofType(Int::class.java).defaultsTo(0)


        // Help option
        acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()


        // Configure settings folder
        val settingsRoot: Path = Paths.get(System.getProperty("user.home", ""), ".pubtrends")
        check(Files.exists(settingsRoot)) {
            "$settingsRoot should have been created by log4j"
        }
        logger.info("Settings folder $settingsRoot")

        val configPath: Path = settingsRoot.resolve("config.properties")
        if (Files.notExists(configPath)) {
            logger.error("Config file not found, please modify and copy config.properties_example to $configPath")
            System.exit(1)
        }

        val config = Properties().apply {
            load(BufferedReader(FileReader(configPath.toFile())))
        }

        val options = parse(*args)
        if (options.has("help")) {
            System.err.print("Arguments: ")
            System.err.println(Arrays.toString(args))
            printHelpOn(System.err)
            System.exit(0)
        }

        logger.info("Crawling...")

        val resetDatabase = options.has("resetDatabase")
        if (resetDatabase) {
            logger.warn("RESETTING DATABASE")
        }

        logger.info("Init database connection")
        val dbHandler = PostgresqlDatabaseHandler(
                config["url"] as String,
                config["port"] as Int,
                config["database"] as String,
                config["user"] as String,
                config["password"] as String,
                resetDatabase)

        logger.info("Init Pubmed processor")
        val pubmedXMLHandler =
                PubmedXMLHandler(dbHandler, config["parserLimit"] as Int, config["batchSize"] as Int)

        logger.info("Init crawler")
        val crawlerProgress = settingsRoot.resolve("crawler")
        val statsFile = settingsRoot.resolve("tag_stats.csv")
        val collectStats = config["collectStats"] as Boolean
        val pubmedCrawler = PubmedCrawler(pubmedXMLHandler, collectStats, statsFile, crawlerProgress)

        val lastCheckCmd = if (options.has("lastCheck")) options.valueOf("lastCheck") as Long else null
        val lastIdCmd = if (options.has("lastId")) options.valueOf("lastId") as Int else null
        if (options.has("retry")) {
            var retry = 1
            logger.info("Retry options presents, keep retrying downloading after any problems.")
             while (pubmedCrawler.update(lastCheckCmd, lastIdCmd)) {
                 logger.info("Waiting for one minute...")
                 Thread.sleep(60*10*1000)
                 logger.info("Retry #$retry")
                 retry += 1
             }
        } else {
            pubmedCrawler.update(lastCheckCmd, lastIdCmd)
        }
        logger.info("Done crawling.")
    }
}