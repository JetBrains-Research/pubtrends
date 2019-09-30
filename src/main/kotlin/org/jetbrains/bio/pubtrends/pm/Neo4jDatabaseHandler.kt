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
        val TRANSACTION_CONFIG = TransactionConfig.builder().withTimeout(Duration.ofSeconds(300)).build()!!
        val RESET_TRANSACTION_CONFIG = TransactionConfig.builder().withTimeout(Duration.ofMinutes(300)).build()!!
        const val PUBLICATION_LABEL = "Publication"
        const val AUTHOR_LABEL = "Author"
        const val AFFILIATION_LABEL = "Affiliation"
        const val DATABANK_LABEL = "Databank"
        const val JOURNAL_LABEL = "Journal"
        const val CITES_LABEL = "CITES"
        const val USES_LABEL = "USES"
        const val WORKS_LABEL = "WORKS_IN"
        const val AUTHORED_LABEL = "AUTHORED"
        const val PUBLISHED_LABEL = "PUBLISHED_IN"
        const val DELETE_BATCH_SIZE = 10000
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
     */
    private fun reset() {
        driver.session().use {
            it.run("CALL apoc.periodic.iterate(\"MATCH (n) RETURN n\", " +
                    "\"DETACH DELETE n\", {batchSize: $DELETE_BATCH_SIZE});", RESET_TRANSACTION_CONFIG)
        }
    }

    private fun initIndexes() {
        driver.session().use {
            it.run("CREATE INDEX ON :$PUBLICATION_LABEL(pmid);")
            it.run("CREATE INDEX ON :$JOURNAL_LABEL(name);")
            it.run("CREATE INDEX ON :$AUTHOR_LABEL(name);")
            it.run("CREATE INDEX ON :$AFFILIATION_LABEL(name);")
            it.run("CREATE INDEX ON :$DATABANK_LABEL(name, accessionNumber)")
        }
    }

    override fun store(articles: List<PubmedArticle>) {
        // Extract relationships
        val articleCitations = articles.map { it.citationList.toSet().map { cit -> it.pmid to cit } }.flatten()
        val articleAuthors = articles.map { it.auxInfo.authors.map { author -> author.name to it.pmid } }.flatten()
        val articleDatabanks = articles.map { it.auxInfo.databanks.map { db -> it.pmid to db }}.flatten()
        val articleJournal = articles.associate { Pair(it.pmid, it.auxInfo.journal.name) }
        val authorAffiliations = articles.map { it.auxInfo.authors }.flatten().toSet().map {
            it.affiliation.map { aff -> it.name to aff }
        }.flatten()

        // Prepare query parameters
        val articleParameters = mapOf("articles" to articles.map { it.toNeo4j() })
        val citationParameters = mapOf("citations" to articleCitations.map { ref ->
            mapOf("pmid_out" to ref.first, "pmid_in" to ref.second)
        })
        val authorParameters = mapOf("authors" to articleAuthors.map { auth ->
            mapOf("name" to auth.first, "pmid" to auth.second)
        })
        val affiliationParameters = mapOf("affiliations" to authorAffiliations.map { aff ->
            mapOf("name" to aff.first, "affiliation" to aff.second)
        })
        val databanksParameters = mapOf("databanks" to articleDatabanks.map { db ->
            mapOf("pmid" to db.first, "name" to db.second.name, "accessionNumber" to db.second.accessionNumber)
        })
        val journalParameters = mapOf("journals" to articleJournal.map { pub ->
            mapOf("pmid" to pub.key, "name" to pub.value)
        })

        driver.session().use {
            // Create new or update existing Publication nodes
            it.run("UNWIND {articles} AS data " +
                    "MERGE (n:$PUBLICATION_LABEL { pmid: toInteger(data.pmid) }) " +
                    "ON CREATE SET n.date = datetime(n.date), n.title = data.title, " +
                    "              n.abstract = data.abstract, n.doi = data.doi, " +
                    "              n.keywords = data.keywords, n.mesh = data.mesh, " +
                    "              n.language = data.language " +
                    "ON MATCH SET n.date = datetime(n.date), n.title = data.title, " +
                    "             n.abstract = data.abstract, n.doi = data.doi, " +
                    "             n.keywords = data.keywords, n.mesh = data.mesh, " +
                    "             n.language = data.language " +
                    "WITH n, data " +
                    "CALL apoc.create.addLabels(id(n), [data.type]) YIELD node " +
                    "RETURN node;",
                    articleParameters, TRANSACTION_CONFIG)

            // Add citation relationships AND create new Publication nodes with pmid only if missing
            it.run("UNWIND {citations} AS cit " +
                    "MATCH (n_out:$PUBLICATION_LABEL { pmid: toInteger(cit.pmid_out) }) " +
                    "MERGE (n_in:$PUBLICATION_LABEL { pmid: toInteger(cit.pmid_in) }) " +
                    "MERGE (n_out)-[:$CITES_LABEL]->(n_in);",
                    citationParameters, TRANSACTION_CONFIG)

            // Add authoring relationships AND create new Author nodes if missing
            it.run("UNWIND {authors} AS auth " +
                    "MATCH (p:$PUBLICATION_LABEL { pmid: toInteger(auth.pmid) }) " +
                    "MERGE (a:$AUTHOR_LABEL { name: auth.name }) " +
                    "MERGE (a)-[:$AUTHORED_LABEL]->(p);",
                    authorParameters, TRANSACTION_CONFIG)

            // Add working relationships AND create new Affiliation nodes if missing
            it.run("UNWIND {affiliations} AS aff " +
                    "MATCH (author:$AUTHOR_LABEL { name: aff.name }) " +
                    "MERGE (affiliation:$AFFILIATION_LABEL { name: aff.affiliation }) " +
                    "MERGE (author)-[:$WORKS_LABEL]->(affiliation);",
                    affiliationParameters, TRANSACTION_CONFIG)

            // Add publishing relationships AND create new Journal nodes if missing
            it.run("UNWIND {journals} AS pub " +
                    "MATCH (n:$PUBLICATION_LABEL { pmid: toInteger(pub.pmid) }) " +
                    "MERGE (j:$JOURNAL_LABEL { name: pub.name }) " +
                    "MERGE (n)-[:$PUBLISHED_LABEL]->(j);",
                    journalParameters, TRANSACTION_CONFIG)

            // Add using relationships AND create new Databank nodes if missing
            it.run("UNWIND {databanks} AS db " +
                    "MATCH (p:$PUBLICATION_LABEL { pmid: toInteger(db.pmid) }) " +
                    "MERGE (d:$DATABANK_LABEL { name: db.name, accessionNumber: db.accessionNumber }) " +
                    "MERGE (p)-[:$USES_LABEL]->(d);",
                    databanksParameters, TRANSACTION_CONFIG)
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
                    "MATCH (n:Publication {pmid: pmid})\n" +
                    "DETACH DELETE n;", deleteParameters, TRANSACTION_CONFIG)
        }
    }

    override fun close() {
        driver.close()
    }
}