package org.jetbrains.bio.pubtrends.pm

import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase


/**
 * This class is used to verify data which was stored in the database.
 */
class TestNeo4jDBHandler(
        url: String,
        port: Int,
        user: String,
        password: String,
        resetDatabase: Boolean
) : Neo4jDatabaseHandler(url, port, user, password, resetDatabase) {

    private val driver = GraphDatabase.driver("bolt://$url:$port", AuthTokens.basic(user, password))

    init {
        if (resetDatabase) {
            reset()
        }
    }

    val articlesCount
        get() = driver.session().use {
            it.run("MATCH (p:Publication) RETURN COUNT(p) AS count;").single()["count"].asInt()
        }

    val articlesPMIDList
        get() = driver.session().use { session ->
            session.run("MATCH (p:Publication) RETURN p.pmid ORDER BY p.pmid ASC;").list().map {
                it["p.pmid"].asInt()
            }
        }
}