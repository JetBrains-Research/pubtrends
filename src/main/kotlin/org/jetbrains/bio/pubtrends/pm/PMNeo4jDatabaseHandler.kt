package org.jetbrains.bio.pubtrends.pm

import com.google.gson.GsonBuilder
import org.jetbrains.bio.pubtrends.AbstractDBHandler
import org.jetbrains.bio.pubtrends.Neo4jConnector
import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import java.io.Closeable

/**
 * See pm_database_supplier.py and pm_loader.py for (up)loading code.
* TODO[shpynov] Consider refactoring.
 */
open class PMNeo4jDatabaseHandler(
        host: String,
        port: Int,
        user: String,
        password: String
) : Neo4jConnector(host, port, user, password), AbstractDBHandler<PubmedArticle> {

    companion object {
        const val DELETE_BATCH_SIZE = 10000
    }

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
                            "INDEX ON :PMPublication(pmid)" }) {
                if (!createOrDelete) {
                    session.run("DROP INDEX ON :PMPublication(pmid)")
                }
            } else if (createOrDelete) {
                session.run("CREATE INDEX ON :PMPublication(pmid)")
            }
            // full text search index
            if (indexes.any { it["description"].toString().trim('"') ==
                            "INDEX ON NODE:PMPublication(title, abstract)" }) {
                if (!createOrDelete) {
                    session.run("CALL db.index.fulltext.drop(\"pmTitlesAndAbstracts\")")
                }
            } else if (createOrDelete) {
                session.run("""
CALL db.index.fulltext.createNodeIndex("${"pmTitlesAndAbstracts"}", ["PMPublication"], ["title", "abstract"])
""".trimIndent())

            }
        }
    }


    override fun store(articles: List<PubmedArticle>) {
        // Prepare queries parameters
        val articleParameters = mapOf("articles" to articles.map {
            mapOf(
                    "pmid" to it.pmid.toString(),
                    "title" to it.title.replace('\n', ' '),
                    "abstract" to it.abstractText.replace('\n', ' '),
                    "date" to (it.date?.toString() ?: ""),
                    "type" to it.type.name,
                    "aux" to GsonBuilder().create().toJson(it.auxInfo)
            )
        })
        val citationParameters = mapOf("citations" to articles.flatMap {
            it.citationList.toSet().map {
                cit -> mapOf("pmid_out" to it.pmid, "pmid_in" to cit) }
        })

        driver.session().use {
            it.run("""
UNWIND {articles} AS data 
MERGE (n:PMPublication { pmid: toInteger(data.pmid) }) 
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
            it.run("""
UNWIND {citations} AS cit 
MATCH (n_out:PMPublication { pmid: toInteger(cit.pmid_out) })
MERGE (n_in:PMPublication { pmid: toInteger(cit.pmid_in) })
MERGE (n_out)-[:PMReferenced]->(n_in);
""".trimIndent(),
                    citationParameters)

        }
    }

    /**
     * This function is used to delete a list of publications with all their relationships.
     *
     * @param ids: list of PMIDs to be deleted
     */
    override fun delete(ids: List<Int>) {
        val deleteParameters = mapOf("pmids" to ids)
        driver.session().use {
            it.run("UNWIND {pmids} AS pmid\n" +
                    "MATCH (n:PMPublication {pmid: toInteger(pmid)})\n" +
                    "DETACH DELETE n;", deleteParameters)
        }
    }

    override fun close() {
        driver.close()
    }

}