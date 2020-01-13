package org.jetbrains.bio.pubtrends.ss

import joptsimple.OptionParser
import joptsimple.ValueConversionException
import joptsimple.ValueConverter
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.system.exitProcess

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
        val (config, configPath, _) = Config.load()
        logger.info("Config\n" + BufferedReader(FileReader(configPath.toFile())).use {
            it.readLines().joinToString("\n")
        })


        logger.info("Init Neo4j database connection")
        val dbHandler = SSNeo4jDatabaseHandler(
                config["url"].toString(),
                config["port"].toString().toInt(),
                config["username"].toString(),
                config["password"].toString()
        )

        dbHandler.use {
            if (options.has("resetDatabase")) {
                logger.info("Resetting database")
                dbHandler.resetDatabase()
            }

            if (options.has("fillDatabase")) {
                val file = File(options.valueOf("fillDatabase").toString())
                logger.info("Started parsing articles $file")
                // TODO ss_batch_size?
                ArchiveParser(dbHandler, file, config["pm_batch_size"].toString().toInt()).parse()
                logger.info("Finished parsing articles")
            }
        }

    }
}