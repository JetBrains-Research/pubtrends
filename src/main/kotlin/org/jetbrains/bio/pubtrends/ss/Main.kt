package org.jetbrains.bio.pubtrends.ss

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.StdOutSqlLogger
import org.jetbrains.exposed.sql.addLogger
import org.jetbrains.exposed.sql.transactions.transaction
import java.io.*
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

    val url = config["url"].toString()
    val port = config["port"].toString().toInt()
    val database = config["database"].toString()
    val user = config["username"].toString()
    val password = config["password"].toString()

    Database.connect(
            url = "jdbc:postgresql://$url:$port/$database",
            driver = "org.postgresql.Driver",
            user = user,
            password = password)

    logger.info("Create SS tables")
    val dropTable = false
    val createTable = true

    if (createTable) {
        transaction {
            addLogger(StdOutSqlLogger)
            if (dropTable) {
                SchemaUtils.drop(SSPublications, SSCitations)
                exec("DROP TYPE Source;")
            }

            exec("CREATE TYPE Source AS ENUM ('Nature', 'Arxiv');")
            SchemaUtils.create(SSPublications, SSCitations)
        }
    }
    logger.info("Parse archive to database")

    val ssTSV = settingsRoot.resolve("semantic_scholar_last.tsv")
    var lastSSId = -1
    if (Files.exists(ssTSV)) {
        BufferedReader(FileReader(ssTSV.toFile())).use {
            val chunks = it.readLine().split("\t")
            when {
                chunks.size == 2 && chunks[0] == "lastSSId" -> {
                    lastSSId = chunks[1].toInt()
                }
            }
        }
    }

    File(config["archive_folder_path"]
            .toString()).walk()
            .filter { !it.name.endsWith(".gz") && it.name.startsWith("s2-corpus") }
            .sorted()
            .drop(lastSSId + 1)
            .forEach {
                logger.info("Started parsing articles $it")
                ArchiveParser(it, config["batchSize"].toString().toInt()).parse()
                logger.info("Finished parsing articles $it")
                BufferedWriter(FileWriter(ssTSV.toFile())).use { br ->
                    br.write("lastSSId\t${it.toString().takeLast(2)}")
                }
            }
}