package org.jetbrains.bio.pubtrends.ss

import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.exceptions.ExposedSQLException
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
import kotlin.system.exitProcess

private val SEMANTIC_SCHOLAR_NAME_REGEX = "s2-corpus-(\\d\\d)\\.gz".toRegex()

fun main(args: Array<String>) {
    val logger = LogManager.getLogger("Pubtrends")

    with(OptionParser()) {
        accepts("resetDatabase", "Reset Database")
        accepts("fillDatabase", "Create and fill database with articles")
        accepts("createIndex", "Create Gin Index")

        acceptsAll(listOf("h", "?", "help"), "Show help").forHelp()

        val options = parse(*args)

        if (options.has("help")) {
            printHelpOn(System.err)
            exitProcess(0)
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
            exitProcess(1)
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

        val resetDatabase = options.has("resetDatabase")
        val fillDatabase = options.has("fillDatabase")
        val createGinIndex = options.has("createIndex")

        if (resetDatabase) {
            logger.info("Resetting database...")

            transaction {
                addLogger(StdOutSqlLogger)

                SchemaUtils.drop(SSPublications, SSCitations)
                try {
                    exec("DROP TYPE Source;")
                } catch (e: ExposedSQLException) {
                    logger.info("Type Source doesn't exist, skipping")
                }
            }
        }

        if (fillDatabase) {
            logger.info("Create tables for Semantic Scholar Database")
            transaction {
                try {
                    exec("CREATE TYPE Source AS ENUM ('Nature', 'Arxiv');")
                } catch (e: ExposedSQLException) {
                    logger.info("Type Source already exists, skipping")
                }
            }

            transaction {
                addLogger(StdOutSqlLogger)

                SchemaUtils.create(SSPublications, SSCitations)
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

            // Replace '~' with '/home/<username>' if path starts with '~'
            val archivePath = config["ss_archive_folder_path"].toString()
                .replace("^~".toRegex(), System.getProperty("user.home"))
            val files = File(archivePath).walk()
                    .filter { SEMANTIC_SCHOLAR_NAME_REGEX.matches(it.name) }
                    .sorted()
                    .drop(lastSSId + 1)
            val filesAmount = files.toList().size

            files.forEachIndexed{index, file ->
                        val id = SEMANTIC_SCHOLAR_NAME_REGEX.matchEntire(file.name)!!.groups[1]!!.value
                        logger.info("Started parsing articles $file")
                        ArchiveParser(file, config["pm_batch_size"].toString().toInt(), curFile = index + 1, filesAmount = filesAmount).parse()
                        logger.info("Finished parsing articles $file")
                        BufferedWriter(FileWriter(ssTSV.toFile())).use { br ->
                            br.write("lastSSId\t$id")
                        }
                    }
        }

        if (createGinIndex) {
            val min = Int.MIN_VALUE
            val max = Int.MAX_VALUE
            val batchSize = 1048576
            val numberOfBatches = (Int.MAX_VALUE.toLong() - Int.MIN_VALUE.toLong()) / batchSize
            var curBatch = 1

            transaction {
                addLogger(StdOutSqlLogger)

                try {
                    exec("alter table sspublications add column if not exists tsv tsvector;")
                } catch (e: ExposedSQLException) {
                    logger.error("Database isn't filled, please launch Semantic Scholar Processor with option 'fillDatabase' at first")
                    exitProcess(1)
                }
            }

            var curStart = min.toLong()
            while (curStart <= max) {
                val curEnd = minOf(curStart + batchSize, max.toLong())
                transaction {
                    exec("""
                    update sspublications
                    set tsv = to_tsvector('english', coalesce(title,'') || coalesce(abstract,''))
                    where crc32id between $curStart and $curEnd;
                    commit;
                    """)
                    logger.info("Added $curBatch/$numberOfBatches batches to tsvector")
                }
                curStart += batchSize
                curBatch++
            }

            logger.info("Creating gin index...")
            transaction {
                addLogger(StdOutSqlLogger)
                val sqlIndexCreation = "create index if not exists pub_gin_index on sspublications using GIN(tsv);"
                exec(sqlIndexCreation)
            }
        }
    }
}