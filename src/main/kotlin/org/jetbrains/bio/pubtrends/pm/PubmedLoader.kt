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

            dbWriter.use {
                if (options.has("resetDatabase")) {
                    LOG.info("Resetting database")
                    dbWriter.reset()
                    Files.deleteIfExists(pubmedLastIdFile)
                    Files.deleteIfExists(pubmedStatsFile)
                }

                if (options.has("fillDatabase")) {
                    LOG.info("Checking Pubmed FTP...")
                    var retry = 1
                    var waitTime: Long = 1
                    var isUpdateRequired = true
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

                            isUpdateRequired = if (retry == 1) {
                                pubmedCrawler.update(lastIdCmd)
                            } else {
                                pubmedCrawler.update(null)
                            }
                            waitTime = 1
                        } catch (e: PubmedCrawlerException) {
                            LOG.error("Error", e)
                            isUpdateRequired = true

                            LOG.info("Waiting for $waitTime seconds...")
                            Thread.sleep(waitTime * 1000)
                            LOG.info("Retry #$retry")
                            retry += 1

                            if (waitTime < 1024) {
                                waitTime *= 2
                            }
                        }
                    }
                    LOG.info("Done crawling.")
                }
            }
        }
    }
}
