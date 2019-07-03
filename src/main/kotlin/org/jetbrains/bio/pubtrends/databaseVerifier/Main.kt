package org.jetbrains.bio.pubtrends.databaseVerifier

//import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.crawler.Citations
import org.jetbrains.bio.pubtrends.crawler.PostgresqlDatabaseHandler
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.transactions.transaction
import java.io.BufferedReader
import java.io.FileReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*

fun main() {

    val logger = LogManager.getLogger("Pubtrends")

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


    logger.info("Init database connection")
    val resetDatabase = false
    val dbHandler = PostgresqlDatabaseHandler(
            config["url"].toString(),
            config["port"].toString().toInt(),
            config["database"].toString(),
            config["username"].toString(),
            config["password"].toString(),
            resetDatabase)

    logger.info("Create extra tables")


    transaction {
        addLogger(StdOutSqlLogger)
        SchemaUtils.create(SemanticScholarCitations, idMatch, PmidCitationsFromSS)
    }

    logger.info("Parse archive to database")

    ArchiveParser(config["archive_path"].toString()).parse()

    addPmidCitations() // in progress

    DatabaseComparator().compareTables(Citations, PmidCitationsFromSS) //in progress
}