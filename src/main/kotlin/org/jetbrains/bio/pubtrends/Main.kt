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
        accepts("lastId", "LastID").withRequiredArg().ofType(Int::class.java).defaultsTo(0)

        // Help option
        acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()

        logger.info("Arguments: \n" + Arrays.toString(args))

        val options = parse(*args)
        if (options.has("help")) {
            printHelpOn(System.err)
            System.exit(0)
        }

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
        logger.info("Config\n" + BufferedReader(FileReader(configPath.toFile())).use {
            it.readLines().joinToString("\n")
        })

        logger.info("Crawling...")

        val resetDatabase = options.has("resetDatabase")
        if (resetDatabase) {
            logger.warn("RESETTING DATABASE")
        }

        logger.info("Init database connection")
        val dbHandler = PostgresqlDatabaseHandler(
                config["url"].toString(),
                config["port"].toString().toInt(),
                config["database"].toString(),
                config["username"].toString(),
                config["password"].toString(),
                resetDatabase)

        logger.info("Init Pubmed processor")
        val pubmedXMLHandler =
                PubmedXMLHandler(
                        dbHandler,
                        config["parserLimit"].toString().toInt(),
                        config["batchSize"].toString().toInt())

        logger.info("Init crawler")
        val crawlerTSV = settingsRoot.resolve("crawler.tsv")
        val statsTSV = settingsRoot.resolve("stats.tsv")
        val collectStats = config["collectStats"].toString().toBoolean()
        val pubmedCrawler = PubmedCrawler(pubmedXMLHandler, collectStats, statsTSV, crawlerTSV)

        val lastIdCmd = if (options.has("lastId")) options.valueOf("lastId").toString().toInt() else null
        if (options.has("retry")) {
            var retry = 1
            logger.info("Retry options presents, keep retrying downloading after any problems.")
            while (pubmedCrawler.update(lastIdCmd)) {
                logger.info("Waiting for 5 minutes...")
                Thread.sleep(5 * 60 * 1000)
                logger.info("Retry #$retry")
                retry += 1
            }
        } else {
            pubmedCrawler.update(lastIdCmd)
        }
        logger.info("Done crawling.")
    }
}