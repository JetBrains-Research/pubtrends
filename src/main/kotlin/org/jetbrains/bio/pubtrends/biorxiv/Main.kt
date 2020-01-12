package org.jetbrains.bio.pubtrends.biorxiv

import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import java.io.BufferedReader
import java.io.FileReader

fun main() {
    val logger = LogManager.getLogger("Pubtrends")

    val scraper = BiorxivScraper()

    // Load configuration file
    val (config, configPath, _) = Config.load()
    logger.info("Config\n" + BufferedReader(FileReader(configPath.toFile())).use {
        it.readLines().joinToString("\n")
    })

    logger.info("Init Neo4j database connection")
    val dbHandler = BiorxivNeo4jDatabaseHandler(
            config["neo4jurl"].toString(),
            config["neo4jport"].toString().toInt(),
            config["neo4jusername"].toString(),
            config["neo4jpassword"].toString()
    )

    logger.info("Extracting links to articles")
    val articleLinks = scraper.extractArticleLinks(10)
    logger.info("Total links extracted: ${articleLinks.size}")

    val articles = articleLinks.map { scraper.extractArticle(it) }
    dbHandler.store(articles)
}