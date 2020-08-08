package org.jetbrains.bio.pubtrends.db

import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import java.io.Closeable

abstract class Neo4jConnector(
        host: String,
        port: Int,
        user: String,
        password: String
): Closeable {

    // Driver objects should be created with application-wide lifetime
    protected val driver = GraphDatabase.driver("bolt://$host:$port", AuthTokens.basic(user, password)).apply {
        try {
            session().use {
                it.run("Match () Return 1 Limit 1")
            }
        } catch (e: Exception) {
            throw IllegalStateException("Failed to connect to Neo4j database. Check config.")
        }
    }

    override fun close() {
        driver.close()
    }

}