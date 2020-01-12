package org.jetbrains.bio.pubtrends.biorxiv

import org.jetbrains.bio.pubtrends.AbstractDBHandler
import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import java.io.Closeable

/**
 * This is a handler to ensure loading the same data structure as stored in PostgreSQL.
 * See pm_loader.py for loading code.
 * TODO[shpynov]: this class shares some code with pm_test_database_loader.py. Consider refactoring.
 */
open class BiorxivNeo4jDatabaseHandler(
        url: String,
        port: Int,
        user: String,
        password: String
) : AbstractDBHandler<BiorxivArticle>, Closeable {

    companion object {
        const val DELETE_BATCH_SIZE = 10000
    }

    // Driver objects should be created with application-wide lifetime
    private val driver = GraphDatabase.driver("bolt://$url:$port", AuthTokens.basic(user, password))

    init {
        processIndexes(true)
    }


    /**
     * This function can be used to wipe contents of the database.
     */
    fun resetDatabase() {
        processIndexes(false)

        driver.session().use {
            // Clear references
            it.run("""
CALL apoc.periodic.iterate("MATCH ()-[r:PMReferenced]->() RETURN r", 
    "DETACH DELETE r", {batchSize: $DELETE_BATCH_SIZE});""".trimIndent())
            // Clear publications
            it.run("""
CALL apoc.periodic.iterate("MATCH (p:PMPublication) RETURN p", 
    "DETACH DELETE p", {batchSize: $DELETE_BATCH_SIZE});""".trimIndent())
        }
    }

    /**
     * @param createOrDelete true to ensure indexes created, false to delete
     */
    private fun processIndexes(createOrDelete: Boolean) {
        driver.session().use { session ->

            // index by pmid
            val indexes = session.run("CALL db.indexes()").list()
            if (indexes.any { it["description"].toString().trim('"') ==
                            "INDEX ON :BiorxivPublication(biorxivId)" }) {
                if (!createOrDelete) {
                    session.run("DROP INDEX ON :BiorxivPublication(biorxivId)")
                }
            } else if (createOrDelete) {
                session.run("CREATE INDEX ON :BiorxivPublication(biorxivId)")
            }
            // full text search index
            if (indexes.any { it["description"].toString().trim('"') ==
                            "INDEX ON NODE:BiorxivPublication(title, abstract)" }) {
                if (!createOrDelete) {
                    session.run("CALL db.index.fulltext.drop(\"biorxivTitlesAndAbstracts\")")
                }
            } else if (createOrDelete) {
                session.run("""
CALL db.index.fulltext.createNodeIndex("${"biorxivTitlesAndAbstracts"}", ["BiorxivPublication"], ["title", "abstract"])
""".trimIndent())

            }
        }
    }


    override fun store(articles: List<BiorxivArticle>) {
        // Prepare queries parameters
        val articleParameters = mapOf("articles" to articles.map { it.toNeo4j() })

        driver.session().use {
            it.run("""
UNWIND {articles} AS data 
MERGE (n:BiorxivPublication { biorxivId: data.biorxivId }) 
ON CREATE SET 
    n.version = data.version,
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.doi = data.doi,
    n.pdfUrl = data.pdfUrl
ON MATCH SET
    n.version = data.version,
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.doi = data.doi,
    n.pdfUrl = data.pdfUrl
RETURN n;
""".trimIndent(),
                    articleParameters)
        }
    }

    /**
     * This function is used to delete a list of publications with all their relationships.
     *
     * @param ids: list of PMIDs to be deleted
     */
    override fun delete(ids: List<Int>) {
        val deleteParameters = mapOf("ids" to ids)
        driver.session().use {
            it.run("UNWIND {ids} AS biorxivId\n" +
                    "MATCH (n:BiorxivPublication {biorxivId: biorxivId})\n" +
                    "DETACH DELETE n;", deleteParameters)
        }
    }

    override fun close() {
        driver.close()
    }

}