package org.jetbrains.bio.pubtrends.ss

import com.google.gson.GsonBuilder
import org.jetbrains.bio.pubtrends.AbstractDBHandler
import org.joda.time.DateTime
import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import java.io.Closeable

/**
* See ss_database_supplier.py and ss_loader.py for (up)loading code.
* TODO[shpynov] Consider refactoring.
 */
open class SSNeo4jDatabaseHandler(
        url: String,
        port: Int,
        user: String,
        password: String
) : AbstractDBHandler<SemanticScholarArticle>, Closeable {

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
CALL apoc.periodic.iterate("MATCH ()-[r:SSReferenced]->() RETURN r", 
    "DETACH DELETE r", {batchSize: $DELETE_BATCH_SIZE});""".trimIndent())
            // Clear publications
            it.run("""
CALL apoc.periodic.iterate("MATCH (p:SSPublication) RETURN p", 
    "DETACH DELETE p", {batchSize: $DELETE_BATCH_SIZE});""".trimIndent())
        }
    }

    /**
     * @param createOrDelete true to ensure indexes created, false to delete
     */
    private fun processIndexes(createOrDelete: Boolean) {
        driver.session().use { session ->

            // index by crc32id
            val indexes = session.run("CALL db.indexes()").list()
            if (indexes.any { it["description"].toString().trim('"') ==
                            "INDEX ON :SSPublication(crc32id)" }) {
                if (!createOrDelete) {
                    session.run("DROP INDEX ON :SSPublication(crc32id)")
                }
            } else if (createOrDelete) {
                session.run("CREATE INDEX ON :SSPublication(crc32id)")
            }
            // full text search index
            if (indexes.any { it["description"].toString().trim('"') ==
                            "INDEX ON NODE:SSPublication(title, abstract)" }) {
                if (!createOrDelete) {
                    session.run("""CALL db.index.fulltext.drop("ssTitlesAndAbstracts")""")
                }
            } else if (createOrDelete) {
                session.run("""
CALL db.index.fulltext.createNodeIndex("ssTitlesAndAbstracts", ["SSPublication"], ["title", "abstract"])
""".trimIndent())

            }
        }
    }


    override fun store(articles: List<SemanticScholarArticle>) {
        // Prepare queries parameters
        val articleParameters = mapOf("articles" to articles.map {
            mapOf(
                    "ssid" to it.ssid,
                    "crc32id" to crc32id(it.ssid).toString(), // Lookup index
                    "pmid" to it.pmid?.toString(),
                    "title" to it.title.replace('\n', ' '),
                    "abstract" to it.abstract?.replace('\n', ' '),
                    "date" to DateTime(it.year ?:1970, 1, 1, 12, 0).toString(),
                    "aux" to GsonBuilder().create().toJson(it.aux)
            )
        })
        val citationParameters = mapOf("citations" to articles.flatMap {
            it.citationList.toSet().map {
                cit -> mapOf(
                    "ssid_out" to it.ssid,
                    "crc32id_out" to crc32id(it.ssid).toString(),
                    "ssid_in" to cit,
                    "crc32id_in" to crc32id(cit).toString())
            }
        })

        driver.session().use {
            it.run("""
UNWIND {articles} AS data 
MERGE (n:SSPublication { crc32id: toInteger(data.crc32id), ssid: data.ssid }) 
ON CREATE SET 
    n.pmid = data.pmid,
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.aux = data.aux
ON MATCH SET 
    n.pmid = data.pmid,
    n.title = data.title,
    n.abstract = data.abstract,
    n.date = datetime(data.date),
    n.aux = data.aux
RETURN n;
""".trimIndent(),
                    articleParameters)

            // Add citation relationships AND create new Publication nodes with ssid only if missing
            it.run("""
UNWIND {citations} AS cit 
MATCH (n_out:SSPublication { crc32id: toInteger(cit.crc32id_out), ssid: cit.ssid_out })
MERGE (n_in:SSPublication { crc32id: toInteger(cit.crc32id_in), ssid: cit.ssid_in })
MERGE (n_out)-[:SSReferenced]->(n_in);
""".trimIndent(),
                    citationParameters)

        }
    }

    override fun close() {
        driver.close()
    }

    override fun delete(ids: List<Int>) {
        throw IllegalStateException("delete is not supported")
    }

}