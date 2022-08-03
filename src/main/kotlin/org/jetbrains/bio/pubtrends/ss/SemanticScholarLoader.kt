package org.jetbrains.bio.pubtrends.ss

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.LoggerContext
import joptsimple.OptionParser
import joptsimple.ValueConversionException
import joptsimple.ValueConverter
import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.db.AbstractDBWriter
import org.jetbrains.bio.pubtrends.db.SemanticScholarPostgresWriter
import org.jetbrains.exposed.sql.Slf4jSqlDebugLogger
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.system.exitProcess

object SemanticScholarLoader {
    private val LOG = LoggerFactory.getLogger(SemanticScholarLoader::class.java)

    @JvmStatic
    fun main(args: Array<String>) {

        with(OptionParser()) {
            accepts("resetDatabase", "Reset Database")
            accepts("finish", "Finish database loading, update indexes")
            accepts("fillDatabase", "Create and fill database with articles")
                .withRequiredArg()
                .withValuesConvertedBy(exists())

            acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()
            acceptsAll(listOf("d", "debug"), "Debug")
            val options = parse(*args)

            if (options.has("help")) {
                printHelpOn(System.err)
                exitProcess(0)
            }
            if (options.has("debug")) {
                val loggerContext = LoggerFactory.getILoggerFactory() as LoggerContext
                val rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME)
                rootLogger.level = Level.DEBUG
                // Keep info level for transactions logger
                loggerContext.getLogger(Slf4jSqlDebugLogger::class.java).level = Level.INFO
            }

            // Load configuration file
            val (config, configPath, settingsRoot) = Config.load()
            LOG.info("Config path: $configPath")

            val dbWriter: AbstractDBWriter<SemanticScholarArticle>
            if (!(config["postgres_host"]?.toString()).isNullOrBlank()) {
                LOG.info("Init Postgresql database connection")
                dbWriter = SemanticScholarPostgresWriter(
                    config["postgres_host"]!!.toString(),
                    config["postgres_port"]!!.toString().toInt(),
                    config["postgres_database"]!!.toString(),
                    config["postgres_username"]!!.toString(),
                    config["postgres_password"]!!.toString(),
                    "finish" in config
                )
            } else {
                throw IllegalStateException("No database configured")
            }


            dbWriter.use {
                if (options.has("resetDatabase")) {
                    LOG.info("Resetting database")
                    dbWriter.reset()
                }

                if (options.has("fillDatabase")) {
                    val statsFile = settingsRoot.resolve("semantic_scholar_stats.tsv")
                    val collectStats = config["loader_collect_stats"].toString().toBoolean()

                    val file = File(options.valueOf("fillDatabase").toString())
                    LOG.info("Started parsing articles $file")
                    SemanticScholarArchiveParser(
                        dbWriter, file,
                        config["loader_batch_size"].toString().toInt(),
                        collectStats,
                        statsFile
                    ).parse()
                    LOG.info("Finished parsing articles")
                }
            }

        }
    }
}

fun exists() = object : ValueConverter<Path> {
    @Throws(ValueConversionException::class)
    override fun convert(value: String): Path {
        val path = Paths.get(value).toAbsolutePath()
        if (!path.toFile().exists()) {
            throw ValueConversionException("Path $path does not exists")
        }
        return path
    }

    override fun valueType() = Path::class.java

    override fun valuePattern(): String? = null
}
