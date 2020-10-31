package examples

import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.arxiv.ArxivCollector
import org.jetbrains.bio.pubtrends.neo4j.DatabaseHandler
import org.jetbrains.bio.pubtrends.validation.ArxivValidator
import org.jetbrains.bio.pubtrends.validation.CrossRefValidator

fun main() {
    val START_DATE = "2020-05-11"
    val dataBaseHandler = DatabaseHandler(
        Config.config["neo4j_url"].toString(),
        Config.config["neo4j_port"].toString(),
        Config.config["neo4j_user"].toString(),
        Config.config["neo4j_password"].toString()
    )
    ArxivCollector.collect(START_DATE, dataBaseHandler, listOf(CrossRefValidator, ArxivValidator))
    dataBaseHandler.close()
}