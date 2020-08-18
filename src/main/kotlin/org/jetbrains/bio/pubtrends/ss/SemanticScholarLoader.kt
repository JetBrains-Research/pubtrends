package org.jetbrains.bio.pubtrends.ss

import joptsimple.OptionParser
import joptsimple.ValueConversionException
import joptsimple.ValueConverter
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.db.SSNeo4JDatabaseWriter
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.system.exitProcess

object SemanticScholarLoader {
    @JvmStatic
    fun main(args: Array<String>) {
        val logger = LogManager.getLogger("Pubtrends")

        with(OptionParser()) {
            accepts("resetDatabase", "Reset Database")
            accepts("fillDatabase", "Create and fill database with articles")
                    .withRequiredArg()
                    .withValuesConvertedBy(exists())

            acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()

            val options = parse(*args)

            if (options.has("help")) {
                printHelpOn(System.err)
                exitProcess(0)
            }

            // Load configuration file
            val (config, configPath, settingsRoot) = Config.load()
            logger.info("Config path: $configPath")

            logger.info("Init Neo4j database connection")
            val dbHandler = SSNeo4JDatabaseWriter(
                    config["neo4jhost"].toString(),
                    config["neo4jport"].toString().toInt(),
                    config["neo4jusername"].toString(),
                    config["neo4jpassword"].toString()
            )

            dbHandler.use {
                if (options.has("resetDatabase")) {
                    logger.info("Resetting database")
                    dbHandler.reset()
                }

                if (options.has("fillDatabase")) {
                    val statsFile = settingsRoot.resolve("semantic_scholar_stats.tsv")
                    val collectStats = config["loader_collect_stats"].toString().toBoolean()

                    val file = File(options.valueOf("fillDatabase").toString())
                    logger.info("Started parsing articles $file")
                    ArchiveParser(dbHandler, file,
                            config["loader_batch_size"].toString().toInt(),
                            collectStats,
                            statsFile
                            ).parse()
                    logger.info("Finished parsing articles")
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
