package org.jetbrains.bio.pubtrends.pm

import org.neo4j.driver.v1.AuthTokens
import org.neo4j.driver.v1.GraphDatabase
import org.neo4j.driver.v1.TransactionConfig
import java.io.Closeable
import java.time.Duration

class Neo4jDatabaseHandler(
        url: String,
        port: Int,
        user: String,
        password: String,
        resetDatabase: Boolean
) : AbstractDBHandler, Closeable {

    companion object {
        val TRANSACTION_CONFIG = TransactionConfig.builder().withTimeout(Duration.ofSeconds(3)).build()!!
        val PUB_LABEL = "Publication"
        val JOURNAL_LABEL = "Journal"
        val CIT_LABEL = "CITES"
        val PUBLISHED_LABEL = "PUBLISHED_IN"
    }

    // Driver objects should be created with application-wide lifetime
    private val driver = GraphDatabase.driver("bolt://$url:$port", AuthTokens.basic(user, password))

    init {
        if (resetDatabase) {
            reset()
        }
        initIndexes()
    }

    /**
     * This function can be used to wipe contents of the database.
     * However, if database is too large, it can cause OutOfMemoryError.
     * APOC extension for neo4j can be a possible solution of this problem.
     */
    private fun reset() {
        driver.session().use {
            it.run("CALL apoc.periodic.iterate(\"MATCH (n) RETURN n\", " +
                    "\"DETACH DELETE n\", {batchSize: 10000});", TRANSACTION_CONFIG)
        }
    }

    private fun initIndexes() {
        driver.session().use {
            it.run("CREATE INDEX ON :$PUB_LABEL(pmid);")
            it.run("CREATE INDEX ON :$JOURNAL_LABEL(name);")
        }
    }

    override fun store(articles: List<PubmedArticle>) {
        val articleParameters = mapOf("articles" to articles.map { it.toNeo4j() })
        val articleCitations = articles.map { it.citationList.toSet().map { cit -> it.pmid to cit } }.flatten()
        val articleJournal = articles.associate { Pair(it.pmid, it.auxInfo.journal.name) }
        val citationParameters = mapOf("citations" to articleCitations.map { ref ->
            mapOf("pmid_out" to ref.first, "pmid_in" to ref.second)
        })
        val journalParameters = mapOf("journals" to articleJournal.map { pub ->
            mapOf("pmid" to pub.key, "name" to pub.value)
        })
        driver.session().use {
            // Create new or update existing Publication nodes
            it.run("UNWIND {articles} AS data " +
                    "MERGE (n:$PUB_LABEL { pmid: toInteger(data.pmid) }) " +
                    "ON CREATE SET n.date = datetime(n.date), n.title = data.title, " +
                    "              n.abstract = data.abstract, n.doi = data.doi, n.language = data.language " +
                    "ON MATCH SET n.date = datetime(n.date), n.title = data.title, " +
                    "             n.abstract = data.abstract, n.doi = data.doi " +
                    "WITH n, data " +
                    "CALL apoc.create.addLabels(id(n), [data.type]) YIELD node " +
                    "RETURN node;",
                    articleParameters, TRANSACTION_CONFIG)

            // Add citation relationships AND create new Publication nodes with pmid only if missing
            it.run("UNWIND {citations} AS cit " +
                    "MATCH (n_out:$PUB_LABEL { pmid: toInteger(cit.pmid_out) }) " +
                    "MERGE (n_in:$PUB_LABEL { pmid: toInteger(cit.pmid_in) }) " +
                    "MERGE (n_out)-[:$CIT_LABEL]->(n_in);",
                    citationParameters, TRANSACTION_CONFIG)

            // Add publishing relationships AND create new Journal nodes if missing
            it.run("UNWIND {journals} AS pub " +
                    "MATCH (n:$PUB_LABEL { pmid: toInteger(pub.pmid) }) " +
                    "MERGE (j:$JOURNAL_LABEL { name: pub.name }) " +
                    "MERGE (n)-[:$PUBLISHED_LABEL]->(j);",
                    journalParameters, TRANSACTION_CONFIG)
        }
    }

    override fun delete(articlePMIDs: List<Int>) {
        val deleteParameters = mapOf("pmids" to articlePMIDs)
        driver.session().use {
            it.run("UNWIND {pmids} AS pmid\n" +
                    "MATCH (n:Publication {pmid: pmid})\n" +
                    "DETACH DELETE n;", deleteParameters, TRANSACTION_CONFIG)
        }
    }

    override fun close() {
        driver.close()
    }
}