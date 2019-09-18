package org.jetbrains.bio.pubtrends.pm

import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import org.neo4j.driver.v1.TransactionConfig
import java.time.Duration

class Neo4jDatabaseHandler(
    url: String,
    port: Int,
    user: String,
    password: String,
    private val resetDatabase: Boolean

) {
    companion object {
        val TRANSACTION_CONFIG = TransactionConfig.builder().withTimeout(Duration.ofSeconds(3)).build()!!
    }

    // Driver objects should be created with application-wide lifetime
    private val driver = GraphDatabase.driver("bolt://$url:$port", AuthTokens.basic(user, password))

    fun init() {
        val article1 = mapOf("title" to "Article Title 1", "year" to 2013, "pmid" to 1)
        val article2 = mapOf("title" to "Article Title 2", "year" to 2015, "pmid" to 2)
        val articleParameters = mapOf("articles" to listOf(article1, article2))
        driver.session().use {
            val cursor = it.run("UNWIND {articles} AS data CREATE (n:Publication) SET n = data RETURN n;",
                    articleParameters, TRANSACTION_CONFIG)
            cursor.list().forEach { record ->
                println(record.asMap())
            }
        }
    }

    fun close() {
        driver.close()
    }
}