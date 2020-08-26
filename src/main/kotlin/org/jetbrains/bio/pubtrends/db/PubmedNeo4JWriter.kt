package org.jetbrains.bio.pubtrends.db

import com.google.gson.GsonBuilder
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.Driver
import org.neo4j.driver.v1.GraphDatabase
import java.io.Closeable

/**
 * See pm_database_supplier.py and pm_loader.py for (up)loading code.
 * TODO[shpynov] Consider refactoring.
 */
open class PubmedNeo4JWriter(
        host: String,
        port: Int,
        user: String,
        password: String
) : AbstractDBWriter<PubmedArticle>, Closeable {

    companion object {
        const val DELETE_BATCH_SIZE = 10000
    }

    private val driver: Driver = GraphDatabase.driver(
            "bolt://$host:$port", AuthTokens.basic(user, password)
    ).apply {
        session().use {
            it.run("Match () Return 1 Limit 1")
        }
    }

    init {
        // Driver objects should be created with application-wide lifetime

        processIndexes(true)
    }

    /**
     * This function can be used to wipe contents of the database.
     */
    override fun reset() {
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

        processIndexes(true)
    }

    override fun finish() {}

    /**
     * @param createOrDelete true to ensure indexes created, false to delete
     */
    private fun processIndexes(createOrDelete: Boolean) {
        driver.session().use { session ->

            // indexes by pmid and doi
            val indexes = session.run("CALL db.indexes()").list()
            listOf("pmid", "doi").forEach { field ->
                if (indexes.any {
                            it["description"].toString().trim('"') ==
                                    "INDEX ON :PMPublication($field)"
                        }) {
                    if (!createOrDelete) {
                        session.run("DROP INDEX ON :PMPublication($field)")
                    }
                } else if (createOrDelete) {
                    session.run("CREATE INDEX ON :PMPublication($field)")
                }
            }

            // full text search index
            if (indexes.any {
                        it["description"].toString().trim('"') ==
                                "INDEX ON NODE:PMPublication(title, abstract)"
                    }) {
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
                    "abstract" to it.abstract.replace('\n', ' '),
                    "date" to (it.date?.toString() ?: ""),
                    "type" to it.type.name,
                    "keywords" to it.keywords.joinToString(","),
                    "mesh" to it.mesh.joinToString(","),
                    "doi" to it.doi,
                    "aux" to GsonBuilder().create().toJson(it.aux)
            )
        })
        val citationParameters = mapOf("citations" to articles.flatMap {
            it.citations.toSet().map { cit -> mapOf("pmid_out" to it.pmid, "pmid_in" to cit) }
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
    n.doi = data.doi,
    n.aux = data.aux,
    n.keywords = data.keywords,
    n.mesh = data.mesh
ON MATCH SET
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.type = data.type,
    n.doi = data.doi,
    n.aux = data.aux,
    n.keywords = data.keywords,
    n.mesh = data.mesh
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
    override fun delete(ids: List<String>) {
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