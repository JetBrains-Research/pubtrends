package org.jetbrains.bio.pubtrends.pm

import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import java.io.Closeable

/**
 * This is a handler to ensure loading the same data structure as stored in PostgreSQL.
 * It can be useful to compare performance of Neo4j vs PostgreSQL [PostgresqlDatabaseHandler].
 * See pm_loader.py for loading code.
 * TODO[shpynov]: this class shares some code with pm_test_database_loader.py. Consider refactoring.
 */
open class Neo4jDatabaseHandler(
        url: String,
        port: Int,
        user: String,
        password: String
) : AbstractDBHandler, Closeable {

    companion object {
        const val PMPublication = "PMPublication"
        const val PMReferenced = "PMReferenced"
        const val pmTitlesAndAbstracts = "pmTitlesAndAbstracts"

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
CALL apoc.periodic.iterate("MATCH ()-[r:$PMReferenced]->() RETURN r", 
    "DETACH DELETE r", {batchSize: $DELETE_BATCH_SIZE});""".trimIndent())
            // Clear publications
            it.run("""
CALL apoc.periodic.iterate("MATCH (p:$PMPublication) RETURN p", 
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
                            "INDEX ON :$PMPublication(pmid)" }) {
                if (!createOrDelete) {
                    session.run("DROP INDEX ON :$PMPublication(pmid)")
                }
            } else if (createOrDelete) {
                session.run("CREATE INDEX ON :$PMPublication(pmid)")
            }
            // full text search index
            if (indexes.any { it["description"].toString().trim('"') ==
                            "INDEX ON NODE:$PMPublication(title, abstract)" }) {
                if (!createOrDelete) {
                    session.run("CALL db.index.fulltext.drop(\"$pmTitlesAndAbstracts\")")
                }
            } else if (createOrDelete) {
                session.run("""
CALL db.index.fulltext.createNodeIndex("$pmTitlesAndAbstracts", ["$PMPublication"], ["title", "abstract"])
""".trimIndent())

            }
        }
    }


    override fun store(articles: List<PubmedArticle>) {
        // Prepare queries parameters
        val articleParameters = mapOf("articles" to articles.map { it.toNeo4j() })
        val citationParameters = mapOf("citations" to articles.flatMap {
            it.citationList.toSet().map {
                cit -> mapOf("pmid_out" to it.pmid.toString(), "pmid_in" to cit.toString()) }
                // NOTE: use toString on indexes here to preserve compatibility between PM and SS
        })

        driver.session().use {
            // NOTE: don't use toInteger(pmid) for indexes here to preserve compatibility between PM and SS
            it.run("""
UNWIND {articles} AS data 
MERGE (n:$PMPublication { pmid: data.pmid }) 
ON CREATE SET 
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.type = data.type,
    n.aux = data.aux
ON MATCH SET 
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.type = data.type,
    n.aux = data.aux
WITH n, data 
CALL apoc.create.addLabels(id(n), [data.type]) YIELD node 
RETURN node;
""".trimIndent(),
                    articleParameters)

            // Add citation relationships AND create new Publication nodes with pmid only if missing
            // NOTE: don't use toInteger(pmid) for indexes here to preserve compatibility between PM and SS
            it.run("""
UNWIND {citations} AS cit 
MATCH (n_out:$PMPublication { pmid: cit.pmid_out })
MERGE (n_in:$PMPublication { pmid: cit.pmid_in })
MERGE (n_out)-[:$PMReferenced]->(n_in);
""".trimIndent(),
                    citationParameters)

        }
    }

    /**
     * This function is used to delete a list of publications with all their relationships.
     *
     * @param articlePMIDs: list of PMIDs to be deleted
     */
    override fun delete(articlePMIDs: List<Int>) {
        val deleteParameters = mapOf("pmids" to articlePMIDs)
        driver.session().use {
            it.run("UNWIND {pmids} AS pmid\n" +
                    "MATCH (n:$PMPublication {pmid: pmid})\n" +
                    "DETACH DELETE n;", deleteParameters)
        }
    }

    override fun close() {
        driver.close()
    }

}