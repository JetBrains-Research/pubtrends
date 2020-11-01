package examples

import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.arxiv.ArxivCollector
import org.jetbrains.bio.pubtrends.db.ArxivNeo4JWriter
import org.jetbrains.bio.pubtrends.validation.ArxivValidator
import org.jetbrains.bio.pubtrends.validation.CrossRefValidator

fun main() {
    val START_DATE = "2020-05-11"
    val dataBaseHandler = ArxivNeo4JWriter(
            Config.config["neo4j_host"].toString(),
            Config.config["neo4j_port"].toString(),
            Config.config["neo4j_username"].toString(),
            Config.config["neo4j_password"].toString()
    )
    ArxivCollector.collect(START_DATE, dataBaseHandler, listOf(CrossRefValidator, ArxivValidator))
    dataBaseHandler.close()
}