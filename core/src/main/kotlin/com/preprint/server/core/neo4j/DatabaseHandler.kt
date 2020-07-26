package com.preprint.server.core.neo4j

import org.neo4j.driver.AuthTokens
import org.neo4j.driver.GraphDatabase
import com.preprint.server.core.arxiv.ArxivData
import com.preprint.server.core.data.Author
import com.preprint.server.core.data.JournalRef
import com.preprint.server.core.data.Reference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.apache.logging.log4j.kotlin.logger
import org.neo4j.driver.Transaction
import java.io.Closeable
import java.lang.Exception
import java.time.LocalDate
import java.util.concurrent.Executors


/**
 * Used to work with neo4j database
 */
class DatabaseHandler(
    url: String,
    port: String,
    user: String,
    password: String
) : Closeable {

    private val driver = GraphDatabase.driver("bolt://$url:$port", AuthTokens.basic(user, password))
    private val logger = logger()
    private val maxThreads = Config.config["neo4j_max_threads"].toString().toInt()

    /**
     * Create initial setup(create indexes, declares some node's properties unique),
     * if the database is accessed for the first time
     */
    init {
        val existingIndexes = driver.session().use { session ->
            session.run("CALL db.indexes() YIELD name").list().map {
                it["name"].asString()
            }
        }

        initIndexes(existingIndexes)
    }


    /**
     * Stores given records to the database.
     * Creates Publication node(whith Arxiv label) for each given record.
     *
     * Creates(if they are not existed) Author nodes, Journal nodes, Affiliation nodes
     * for each author, journal or affiliation presented in the given data.
     *
     * Also creates(if they are not existed) Publication nodes for validated references
     * For non validated references, that are not found in the database, Missing publication
     * nodes is created
     *
     * And finally creates connections between nodes(connections are created with multiple threads)
     *
     * For now there is no full description of the database(I hope it will appear later),
     * so the only way to understand how it works is to read the code below
     */
    fun storeArxivData(arxivRecords: List<ArxivData>) {
        logger.info("Begin storing ${arxivRecords.size} records to the database")


        /*
        Create 'Publication' nodes with 'Arxiv' label for records from `arxivRecords`
        And store ids(that neo4j gives them) of the created records in `pubIds
        `pubIds` will be later used to create connections between nodes
         */
        var pubIds: List<Long> = listOf()
        driver.session().use {
            //create new or update publication node
            pubIds = mutableListOf()
            it.writeTransaction { tr ->
                val publications = mapOf("publications" to arxivRecords.map { arxivDataToMap(it) })
                pubIds = tr.run(
                    """
                    UNWIND ${"$"}publications as pubData
                    MERGE (pub:${DBLabels.PUBLICATION.str} {arxivId : pubData.arxivId}) 
                    SET pub += pubData, pub:${DBLabels.ARXIV_LBL.str}
                    RETURN id(pub)
                """.trimIndent(), publications
                ).list().map { it.get("id(pub)").asLong() }
            }
        }
        logger.info("Publication nodes created")

        /*
        Gets all nodes from the arxivRecords(authors, journals, etc.) and create nodes for them
        This is done in order to use concurrency later for connection creation
         */
        val newPublications = createAllNodes(arxivRecords)

        /*
        Create all connections using multiple threads,
        But some transaction fail(because of concurrency) and
        we will store them in `failedTransactions` and retry them later sequentially
         */
        val failedTransactions = mutableListOf<Pair<Long, ArxivData>>()
        val dispatcher = if (maxThreads > 0) Executors.newFixedThreadPool(maxThreads).asCoroutineDispatcher()
                         else Dispatchers.Default
        runBlocking(dispatcher) {
            pubIds.zip(arxivRecords).forEach { (id, record) ->
                launch {
                    driver.session().use {
                        try {
                            it.writeTransaction { tr ->
                                try {
                                    createConnections(tr, id, record, newPublications)
                                } catch (e: Exception) {
                                    failedTransactions.add(Pair(id, record))
                                }
                            }
                        } catch (e : Exception) {
                            logger.error("Connections creation failed for ${record.id}")
                        }
                    }
                }
            }
        }
        logger.info("Failed ${failedTransactions.size} transactions")
        logger.info("Retrying failed transactions")
        driver.session().use {session ->
            failedTransactions.forEach { (id, record) ->
                try {
                    session.writeTransaction { tr ->
                        createConnections(tr, id, record, newPublications)
                    }
                } catch (e: Exception) {
                    logger.error("Connections creation failed again for ${record.id}")
                    logger.error(e.message.toString())
                }
            }
        }

        logger.info("All connections created")
    }

    /**
     * Creates indexes for some properties, if they weren't created before.
     * `existingIndexes` contains the name of all indexes presented in the database
     */
    private fun initIndexes(existingIndexes: List<String>) {
        driver.session().use {
            it.writeTransaction { tr ->
                if (!existingIndexes.contains("publication_arxivId")) {
                    tr.run(
                        """CREATE INDEX publication_arxivId FOR (p:${DBLabels.PUBLICATION.str})
                          ON (p.arxivId)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("publication_doi")) {
                    tr.run(
                        """CREATE INDEX publication_doi FOR (p:${DBLabels.PUBLICATION.str})
                          ON (p.doi)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("publication_title")) {
                    tr.run(
                        """CREATE INDEX publication_title FOR (p:${DBLabels.PUBLICATION.str})
                          ON (p.title)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("author_name")) {
                    tr.run(
                        """CREATE INDEX author_name FOR (a:${DBLabels.AUTHOR.str})
                          ON (a.name)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("journal_title")) {
                    tr.run(
                        """CREATE INDEX journal_title FOR (j:${DBLabels.JOURNAL.str})
                          ON (j.title)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("affiliation_name")) {
                    tr.run(
                        """CREATE INDEX affiliation_name FOR (a:${DBLabels.AFFILIATION.str})
                          ON (a.name)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("missing_publication_title")) {
                    tr.run(
                        """CREATE INDEX missing_publication_title FOR (m:${DBLabels.MISSING_PUBLICATION.str})
                          ON (m.title)"""
                            .trimIndent()
                    )
                }

                if (!existingIndexes.contains("full_text_search_publication_index")) {
                    tr.run(
                        """
                            CALL db.index.fulltext.createNodeIndex("full_text_search_publication_index",
                                ["Arxiv"],["title", "arxivId", "doi", "pubYear", "journalVolume",
                                           "journalFirstPage", "journalLastPage", "journalIssue"])
                        """.trimIndent()
                    )
                }

                if (!existingIndexes.contains("full_text_search_abstract_index")) {
                    tr.run(
                        """
                            CALL db.index.fulltext.createNodeIndex("full_text_search_abstract_index",
                                ["Arxiv"],["abstract", "title"])
                        """.trimIndent()
                    )
                }
            }
        }
    }

    /**
     * Create all nodes that needed for the given records(Author, Journal, MissingPublication, etc.)
     * Methods getAll... get all data of the type ...(Author e.g) from given list of records
     * Methods createAll... creates a new nodes in the database for given data of type ...(Author e.g.)
     */
    private fun createAllNodes(arxivRecords: List<ArxivData>): Set<Reference> {
        val authors = getAllAuthors(arxivRecords)
        val journals = getAllJouranls(arxivRecords)
        val affiliations = getAllAffiliations(arxivRecords)
        val mpubs = getAllMissingPublicationsWithTitle(arxivRecords)
        val pubs = getAllPublications(arxivRecords)

        createAllAuthors(authors)
        logger.info("Author nodes created")
        createAllJournals(journals)
        logger.info("Journal nodes created")
        createAllAffiliations(affiliations)
        logger.info("Affiliation nodes created")
        createAllMissingPublicationsWithTitle(mpubs)
        logger.info("MissingPublication nodes created")
        val newPubliactions = createAllPublications(pubs)
        logger.info("Publication nodes created")
        return newPubliactions
    }

    private fun createAllAuthors(authors: List<Author>) {
        val params = mapOf(
            "authors" to authors.map {it.name}
        )

        driver.session().use {
            it.writeTransaction {tr ->
                tr.run("""
                    UNWIND ${parm("authors")} as authorName
                    MERGE (a:${DBLabels.AUTHOR.str} {name : authorName})
                """.trimIndent(), params)
            }
        }
    }

    private fun createAllJournals(journals: List<String>) {
        val params = mapOf(
            "journals" to journals
        )

        driver.session().use {
            it.writeTransaction { tr ->
                tr.run("""
                    UNWIND ${parm("journals")} as journalTitle
                    MERGE (j:${DBLabels.JOURNAL.str} {title : journalTitle})
                """.trimIndent(), params)
            }
        }
    }

    private fun createAllAffiliations(affs: List<String>) {
        val params = mapOf(
            "affs" to affs
        )

        driver.session().use {
            it.writeTransaction { tr ->
                tr.run("""
                    UNWIND ${parm("affs")} as aname
                    MERGE (a:${DBLabels.AFFILIATION.str} {name : aname})
                """.trimIndent(), params)
            }
        }
    }

    private fun createAllMissingPublicationsWithTitle(mpubs: List<Reference>) {
        driver.session().use {
            it.writeTransaction { tr ->
                mpubs.forEach { ref ->
                    val params = mapOf(
                        "doi" to ref.doi,
                        "arxivId" to ref.arxivId,
                        "title" to ref.title
                    )
                    val pubs = tr.run("""
                        MATCH (p:Publication)
                        WHERE p.doi = ${parm("doi")} 
                              OR p.arxivId = ${parm("arxivId")}
                              OR p.title = ${parm("title")}
                        RETURN p
                    """.trimIndent(), params)
                    if (pubs.list().isEmpty()) {
                        tr.run("""
                            MERGE (p:${DBLabels.MISSING_PUBLICATION.str} {title : ${parm("title")}})
                        """.trimIndent(), params)
                    }
                }
            }
        }
    }

    private fun createAllPublications(pubs: List<Reference>): Set<Reference> {
        val newPub = mutableSetOf<Reference>()
        driver.session().use {
            it.writeTransaction { tr ->
                pubs.forEach { ref ->
                    val params = mapOf(
                        "doi" to ref.doi,
                        "arxivId" to ref.arxivId,
                        "title" to ref.title
                    )
                    val resp = tr.run("""
                        MATCH (p:Publication)
                        WHERE p.doi = ${parm("doi")} 
                              OR p.arxivId = ${parm("arxivId")}
                              OR p.title = ${parm("title")}
                        RETURN p
                    """.trimIndent(), params)
                    if (resp.list().isEmpty()) {
                        newPub.add(ref)
                        val matchString =
                            if (!ref.doi.isNullOrEmpty()) "doi: ${parm("doi")}"
                            else if (!ref.arxivId.isNullOrEmpty()) "arxivId: ${parm("arxivId")}"
                            else "title: ${parm("title")}"
                        tr.run("""
                            MERGE (p:${DBLabels.PUBLICATION.str} {${matchString}})
                        """.trimIndent(), params)
                    }
                }
            }
        }
        return newPub
    }

    private fun getAllAuthors(records: List<ArxivData>): List<Author> {
        val authors = mutableSetOf<Author>()
        records.forEach { record ->
            record.authors.forEach { authors.add(it) }
            record.refList.forEach { ref ->
                if (ref.validated) {
                    ref.authors.forEach { authors.add(it) }
                }
            }
        }
        return authors.toList()
    }

    private fun getAllJouranls(records : List<ArxivData>) : List<String> {
        val journals = mutableSetOf<String>()
        records.forEach { record ->
            if (!record.journal?.rawTitle.isNullOrBlank()) {
                journals.add(record.journal!!.rawTitle!!)
            }
            record.refList.forEach { ref ->
                if (ref.validated && !ref.journal.isNullOrBlank()) {
                    journals.add(ref.journal!!)
                }
            }
        }
        return journals.toList()
    }

    private fun getAllAffiliations(records: List<ArxivData>): List <String> {
        val affiliations = mutableSetOf<String>()
        records.forEach {record ->
            record.authors.forEach {
                if (!it.affiliation.isNullOrBlank()) {
                    affiliations.add(it.affiliation)
                }
            }

            record.refList.forEach {ref ->
                if (ref.validated) {
                    ref.authors.forEach { author ->
                        if (!author.affiliation.isNullOrBlank()) {
                            affiliations.add(author.affiliation)
                        }
                    }
                }
            }
        }
        return affiliations.toList()
    }

    private fun getAllMissingPublicationsWithTitle(records: List<ArxivData>): List <Reference> {
        val mpubs = mutableSetOf<Reference>()
        records.forEach { record ->
            record.refList.forEach {ref ->
                if (!ref.validated && !ref.title.isNullOrBlank()) {
                    mpubs.add(ref)
                }
            }
        }
        return mpubs.toList()
    }

    private fun getAllPublications(records: List<ArxivData>): List<Reference> {
        val pubs = mutableSetOf<Reference>()
        records.forEach { record ->
            record.refList.forEach {ref ->
                if (ref.validated) {
                    pubs.add(ref)
                }
            }
        }
        return pubs.toList()
    }

    private fun arxivDataToMap(record: ArxivData): Map<String, Any> {
        val res = mutableMapOf<String, Any>()
        res += "title" to record.title
        res += "arxivId" to record.id
        if (record.doi != null) {
            res += "doi" to record.doi!!
        }
        if (record.mscClass != null) {
            res += "mscClass" to record.mscClass!!
        }
        if (record.acmClass != null) {
            res += "acmClass" to record.acmClass!!
        }
        if (record.categories.isNotEmpty()) {
            res += "categories" to record.categories
        }
        if (!record.creationDate.isBlank()) {
            res += "creationDate" to LocalDate.parse(record.creationDate)
        }
        if (!record.lastUpdateDate.isNullOrBlank()) {
            res += "lastUpdateDate" to LocalDate.parse(record.lastUpdateDate!!)
        }
        if (record.pdfUrl.isNotBlank()) {
            res += "pdfUrl" to record.pdfUrl
        }
        if (record.abstract.isNotBlank()) {
            res += "abstract" to record.abstract
        }
        if (record.journal?.year != null) {
            res += "pubYear" to record.journal!!.year!!
        }
        return res
    }

    private fun refDataToMap(ref: Reference): Map<String, Any> {
        val res = mutableMapOf<String, Any>()
        if (!ref.title.isNullOrEmpty()) {
            res += "title" to ref.title!!
        }
        if (!ref.arxivId.isNullOrEmpty()) {
            res += "arxivId" to ref.arxivId!!
        }
        if (!ref.doi.isNullOrEmpty()) {
            res += "doi" to ref.doi!!
        }
        if (ref.year != null) {
            res += "pubYear" to ref.year!!
        }
        if (ref.pmid != null) {
            res += "pmid" to ref.pmid!!
        }
        if (ref.ssid != null) {
            res += "ssid" to ref.ssid!!
        }
        if (ref.urls.isNotEmpty()) {
            res += "urls" to ref.urls
        }
        return res
    }

    private fun journalDataToMap(ref: Reference): Map<String, Any> {
        val res = mutableMapOf<String, Any>()
        if (!ref.journal.isNullOrEmpty()) {
            res += "jornal" to ref.journal!!
        }
        if (!ref.volume.isNullOrEmpty()) {
            res += "volume" to ref.volume!!
        }
        if (!ref.issue.isNullOrEmpty()) {
            res += "issue" to ref.issue!!
        }
        if (ref.year != null) {
            res += "pubYear" to ref.year!!
        }
        if (ref.firstPage != null) {
            res += "firstPage" to ref.firstPage!!
        }
        if (ref.lastPage != null) {
            res += "lastPage" to ref.lastPage!!
        }
        return res
    }

    /**
     * This function returns a string that can be used as parameter name
     * in cypher queries in mulitiline strings("""...""")
     */
    private fun parm(paramName: String): String {
        return "\$$paramName"
    }


    /**
     * Creates all connections that needed for given record
     */
    private fun createConnections(
            tr : Transaction,
            id : Long,
            record: ArxivData,
            newPublications: Set<Reference>
    ) {
        createAuthorConnections(tr, record.authors, id)

        createCitationsConnections(tr, record, newPublications)

        if (record.journal != null) {
            createJournalPublicationConnections(tr, record.journal, id)
        }
    }

    /**
     * Creates Publication -> Author connection and Author -> Affiliation connections
     */
    private fun createAuthorConnections(
            tr: Transaction,
            authors: List<Author>,
            id: Long
    ) {

        authors.forEach {author ->
            val params = mapOf(
                "name" to author.name,
                "aff" to author.affiliation,
                "pubId" to id
            )
            val createAffiliationQuery =
                if (author.affiliation != null) {
                    """
                        MATCH (aff:${DBLabels.AFFILIATION.str} {name: ${parm("aff")}})
                        MERGE (auth)-[:${DBLabels.WORKS.str}]->(aff)
                    """.trimIndent()
                } else ""

            tr.run("""
                    MATCH (auth:${DBLabels.AUTHOR.str} {name: ${parm("name")}})
                    $createAffiliationQuery
                    WITH auth
                    MATCH (pub:${DBLabels.PUBLICATION.str})
                    WHERE id(pub) = ${parm("pubId")}
                    MERGE (pub)-[:${DBLabels.AUTHORED.str}]->(auth)
                """.trimIndent(), params)
        }
    }

    /**
     * Creates Publication -> Publication connections and MissingPublication nodes
     *
     * More detailed:
     * Let it have a Reference `ref`.
     * It searches Publication node for `ref`(by title, doi or arxivId)
     * in the database and then there is several cases:
     * 1)If finds, then just creates new connection from the node that corresponds `record` to the found node.
     * 2)If doesn't find, but `ref` has a title, than it trying to find a MissingPublication node with this title
     * and creates connection if found, or creates new MissingPublication node
     * 3)Otherwise just creates MissingPublication node.
     *
     * To optimize queries this search was done before(in createAllNodes)
     * and all new Publications are stored in `newPublications`
     */
    private fun createCitationsConnections(
        tr: Transaction,
        record: ArxivData,
        newPublications: Set<Reference>) {
        record.refList.forEach {ref ->
            val params = mapOf(
                "rid" to record.id,
                "arxId" to ref.arxivId,
                "rdoi" to ref.doi,
                "rtit" to ref.title,
                "rRef" to ref.rawReference,
                "cdata" to refDataToMap(ref),
                "jdata" to journalDataToMap(ref)
            )
            if (!newPublications.contains(ref)) {
                val res = tr.run(
                    """
                    MATCH (pubFrom:${DBLabels.PUBLICATION.str} {arxivId: ${parm("rid")}})
                    MATCH (pubTo:${DBLabels.PUBLICATION.str})
                    WHERE pubTo <> pubFrom AND (pubTo.arxivId = ${parm("arxId")} OR
                        pubTo.doi = ${parm("rdoi")} OR pubTo.title = ${parm("rtit")})
                    SET pubTo += ${parm("cdata")}
                    MERGE (pubFrom)-[c:${DBLabels.CITES.str} {rawRef: ${parm("rRef")}}]->(pubTo)
                    RETURN pubTo
                """.trimIndent(), params
                )

                if (res.list().isEmpty()) {
                    if (!ref.title.isNullOrEmpty()) {
                        //then the cited publication doesn't exist in database
                        //crete missing publication -> publication connection
                        tr.run(
                            """
                    MATCH (pub:${DBLabels.PUBLICATION.str} {arxivId: ${parm("rid")}})
                    MATCH (mpub:${DBLabels.MISSING_PUBLICATION.str} {title: ${parm("rtit")}})
                    MERGE (mpub)-[c:${DBLabels.CITED_BY.str}]->(pub)
                    SET c.rawRef = ${parm("rRef")}, 
                        mpub += ${parm("cdata")},
                        mpub += ${parm("jdata")}
                """.trimIndent(), params
                        )
                    } else {
                        tr.run(
                            """	
                    MATCH (pub:${DBLabels.PUBLICATION.str} {arxivId: ${parm("rid")}})	
                    CREATE (mpub:${DBLabels.MISSING_PUBLICATION.str})	
                    MERGE (mpub)-[c:${DBLabels.CITED_BY.str}]->(pub)	
                    SET c.rawRef = ${parm("rRef")}, 	
                        mpub += ${parm("cdata")},	
                        mpub += ${parm("jdata")}	
                """.trimIndent(), params
                        )
                    }
                }
            }
            else {
                val params = mapOf(
                    "cdata" to refDataToMap(ref),
                    "arxivId" to record.id,
                    "rawRef" to ref.rawReference
                )
                val matchString =
                    if (!ref.doi.isNullOrEmpty()) "doi: ${parm("cdata.doi")}"
                    else if (!ref.arxivId.isNullOrEmpty()) "arxivId: ${parm("cdata.arxivId")}"
                    else "title: ${parm("cdata.title")}"

                val idObj = tr.run(
                    """
                        MATCH (pub:${DBLabels.PUBLICATION.str} {arxivId: ${parm("arxivId")}})
                        MATCH (cpub:${DBLabels.PUBLICATION.str} {${matchString}})
                        SET cpub += ${parm("cdata")}
                        MERGE (pub)-[cites:${DBLabels.CITES.str}]->(cpub)
                        SET cites.rawRef = ${parm("rawRef")}
                        RETURN id(cpub)
                        """.trimIndent(),
                    params
                ).list().map { it.get("id(cpub)").asLong() }
                if (idObj.size > 0) {
                    val id = idObj[0]
                    ref.authors.let { createAuthorConnections(tr, it, id) }

                    if (!ref.journal.isNullOrBlank()) {
                        val journal = JournalRef(
                            rawTitle = ref.journal, volume = ref.volume, firstPage = ref.firstPage,
                            lastPage = ref.lastPage, number = ref.issue, issn = ref.issn, rawRef = ""
                        )
                        createJournalPublicationConnections(tr, journal, id)
                    }
                } else {
                    logger.error(
                        "Failed to create connection between Publication with arxivId ${record.id} " +
                                "and publication with match string $matchString and doi ${ref.doi}"
                    )
                }
            }
        }
    }

    /**
     * Create Publication->Journal connection
     */
    private fun createJournalPublicationConnections(
            tr: Transaction,
            journal: JournalRef?,
            id: Long
    ) {

        if (journal?.rawTitle != null) {
            val params = mapOf(
                "pubId" to id,
                "rjrl" to journal.rawTitle,
                "vol" to journal.volume,
                "firstPage" to journal.firstPage,
                "lastPage" to journal.lastPage,
                "no" to journal.number,
                "rr" to journal.rawRef
            )
            tr.run("""
                   MATCH (pub:${DBLabels.PUBLICATION.str})
                   WHERE id(pub) = ${parm("pubId")}
                   MATCH (j:${DBLabels.JOURNAL.str} {title: ${parm("rjrl")}})
                   MERGE (pub)-[jref:${DBLabels.PUBLISHED_IN}]->(j)
                   ON CREATE SET pub.journalVolume = ${parm("vol")}, pub.journalFirstPage = ${parm("firstPage")},
                       pub.journalLastPage = ${parm("lastPage")}, pub.journalIssue = ${parm("no")},
                       jref.rawRef = ${parm("rr")}
                """.trimIndent(), params)
        }
    }

    override fun close() {
        driver.close()
    }
}