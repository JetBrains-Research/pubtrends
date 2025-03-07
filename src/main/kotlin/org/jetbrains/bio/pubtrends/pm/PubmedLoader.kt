package org.jetbrains.bio.pubtrends.pm

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.LoggerContext
import joptsimple.OptionParser
import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.db.AbstractDBWriter
import org.jetbrains.bio.pubtrends.db.PubmedPostgresWriter
import org.slf4j.LoggerFactory
import java.nio.file.Files
import kotlin.system.exitProcess


object PubmedLoader {

    // 1 minute
    private const val WAIT_TIME = 60

    private const val MAX_DB_RETRIES = 3

    private val LOG = LoggerFactory.getLogger(PubmedLoader::class.java)

    @JvmStatic
    fun main(args: Array<String>) {

        with(OptionParser()) {
            accepts("resetDatabase", "Reset Database")
            accepts("fillDatabase", "Create and fill database with articles")

            accepts("lastId", "LastID").withRequiredArg().ofType(Int::class.java).defaultsTo(0)

            // Help option
            acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()
            acceptsAll(listOf("d", "debug"), "Debug")

            LOG.info("Arguments: \n" + args.contentToString())

            val options = parse(*args)
            if (options.has("help")) {
                printHelpOn(System.err)
                exitProcess(0)
            }
            if (options.has("debug")) {
                val loggerContext = LoggerFactory.getILoggerFactory() as LoggerContext
                loggerContext.getLogger(Logger.ROOT_LOGGER_NAME).level = Level.DEBUG
            }

            // Load configuration file
            val (config, configPath, settingsRoot) = Config.load()
            LOG.info("Config path: $configPath")

            val dbWriter: AbstractDBWriter<PubmedArticle>
            if (!(config["postgres_host"]?.toString()).isNullOrBlank()) {
                LOG.info("Init Postgresql database connection")
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

            var dbRetry = 1
            var isUpdateRequired = true
            while (isUpdateRequired) {
                try {
                    dbWriter.use {
                        if (options.has("resetDatabase")) {
                            LOG.info("Resetting database")
                            dbWriter.reset()
                            Files.deleteIfExists(pubmedLastIdFile)
                            Files.deleteIfExists(pubmedStatsFile)
                        }

                        if (options.has("fillDatabase")) {
                            LOG.info("Checking Pubmed FTP...")
                            var downloadRetry = 1

                            LOG.info("Retrying downloading after any problems.")
                            while (isUpdateRequired) {
                                try {
                                    LOG.info("Init Pubmed processor")
                                    val pubmedXMLParser =
                                        PubmedXMLParser(dbWriter, config["loader_batch_size"].toString().toInt())

                                    LOG.info("Init crawler")
                                    val collectStats = config["loader_collect_stats"].toString().toBoolean()
                                    val pubmedCrawler = PubmedCrawler(
                                        pubmedXMLParser, collectStats,
                                        pubmedStatsFile, pubmedLastIdFile
                                    )

                                    val lastIdCmd = if (options.has("lastId"))
                                        options.valueOf("lastId").toString().toInt()
                                    else
                                        null
                                    isUpdateRequired = if (downloadRetry == 1) {
                                        pubmedCrawler.update(lastIdCmd)
                                    } else {
                                        pubmedCrawler.update(null)
                                    }
                                } catch (e: PubmedCrawlerException) {
                                    LOG.error("Error", e)
                                    isUpdateRequired = true

                                    LOG.info("Download error, retrying in $WAIT_TIME seconds...")
                                    Thread.sleep(WAIT_TIME * 1000L)
                                    LOG.info("Download retry #$downloadRetry")
                                    downloadRetry += 1
                                }
                            }
                            LOG.info("Done crawling.")
                        }

                        dbRetry = 1
                    }
                } catch (e: Exception) {
                    LOG.error("Database connection error, retrying in $WAIT_TIME seconds...")
                    dbWriter.close()
                    if (dbRetry > MAX_DB_RETRIES) {
                        LOG.error("Database connection error, maximum retries reached.")
                        exitProcess(1)
                    }
                    Thread.sleep(WAIT_TIME * 1000L)
                    LOG.info("Database connection retry #$dbRetry")
                    dbRetry += 1
                }
            }
        }
    }
}