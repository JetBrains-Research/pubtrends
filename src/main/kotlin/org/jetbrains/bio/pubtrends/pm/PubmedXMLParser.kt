package org.jetbrains.bio.pubtrends.pm

import org.apache.logging.log4j.LogManager
import org.joda.time.DateTime
import org.joda.time.IllegalFieldValueException
import java.io.*
import java.util.zip.GZIPInputStream
import javax.xml.namespace.QName
import javax.xml.stream.XMLEventReader
import javax.xml.stream.XMLInputFactory
import javax.xml.stream.XMLStreamException


class PubmedXMLParser(
        private val dbHandler: AbstractDBHandler,
        private val parserLimit: Int,
        private val batchSize: Int = 0
) {
    companion object {
        private val logger = LogManager.getLogger(PubmedXMLParser::class)

        // Article tag
        const val ARTICLE_TAG = "PubmedArticleSet/PubmedArticle"

        // Sub-article tags
        private const val MEDLINE_CITATION_TAG = "PubmedArticleSet/PubmedArticle/MedlineCitation"
        private const val PUBMED_DATA_TAG = "PubmedArticleSet/PubmedArticle/PubmedData"

        // Tags that actually contain data
        const val PMID_TAG = "$MEDLINE_CITATION_TAG/PMID"
        const val DELETED_PMID_TAG = "PubmedArticleSet/DeleteCitation/PMID"
        const val YEAR_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/Year"
        const val MONTH_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/Month"
        const val DAY_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/Day"
        const val MEDLINE_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/MedlineDate"

        const val PUBLICATION_TYPE_TAG = "$MEDLINE_CITATION_TAG/Article/PublicationTypeList/PublicationType"
        const val TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/ArticleTitle"

        const val ABSTRACT_TAG = "$MEDLINE_CITATION_TAG/Article/Abstract/AbstractText"
        const val OTHER_ABSTRACT_TAG = "$MEDLINE_CITATION_TAG/OtherAbstract/AbstractText"

        const val AUTHOR_TAG = "$MEDLINE_CITATION_TAG/Article/AuthorList/Author"
        const val AUTHOR_LASTNAME_TAG = "$AUTHOR_TAG/LastName"
        const val AUTHOR_INITIALS_TAG = "$AUTHOR_TAG/Initials"
        const val AUTHOR_AFFILIATION_TAG = "$AUTHOR_TAG/AffiliationInfo/Affiliation"

        const val CITATION_PMID_TAG = "$PUBMED_DATA_TAG/ReferenceList/Reference/ArticleIdList/ArticleId"
        const val KEYWORD_TAG = "$MEDLINE_CITATION_TAG/KeywordList/Keyword"

        const val MESH_HEADING_TAG = "$MEDLINE_CITATION_TAG/MeshHeadingList/MeshHeading"
        const val MESH_DESCRIPTOR_TAG = "$MESH_HEADING_TAG/DescriptorName"
        const val MESH_QUALIFIER_TAG = "$MESH_HEADING_TAG/QualifierName"

        const val DATABANK_TAG = "$MEDLINE_CITATION_TAG/Article/DataBankList/DataBank"
        const val DATABANK_NAME_TAG = "$DATABANK_TAG/DataBankName"
        const val ACCESSION_NUMBER_TAG = "$DATABANK_TAG/AccessionNumberList/AccessionNumber"

        const val DOI_TAG = "$PUBMED_DATA_TAG/ArticleIdList/ArticleId"
        const val LANGUAGE_TAG = "$MEDLINE_CITATION_TAG/Article/Language"
        const val JOURNAL_TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/Title"

        // Month mapping
        private val CALENDAR_MONTH = mapOf(
                "Jan" to 1, "Feb" to 2, "Mar" to 3, "Apr" to 4, "May" to 5, "Jun" to 6,
                "Jul" to 7, "Aug" to 8, "Sep" to 9, "Oct" to 10, "Nov" to 11, "Dec" to 12
        )
    }

    private val factory = XMLInputFactory.newFactory()!!

    init {
        factory.setProperty(XMLInputFactory.SUPPORT_DTD, false)
    }

    // Container for parsed articles
    private var articlesStored = 0
    internal val articleList = mutableListOf<PubmedArticle>()
    internal val deletedArticlePMIDList = mutableListOf<Int>()

    // Stats about XML tags
    val tags = HashMap<String, Int>()

    fun parse(name: String): Boolean {
        try {
            val file = File(name)
            logger.debug("File location: ${file.absolutePath}")
            val `in` = if (name.endsWith(".gz"))
                GZIPInputStream(BufferedInputStream(FileInputStream(file)))
            else // for tests
                BufferedInputStream(FileInputStream(file))
            `in`.use {
                parseData(factory.createXMLEventReader(it))
            }
        } catch (e: XMLStreamException) {
            logger.error("Failed to parse $name", e)
            throw e
        }

        return true
    }

    private fun parseData(eventReader: XMLEventReader) {
        // Stats about articles & tags
        var articleCounter = 0
        var keywordCounter = 0
        var citationCounter = 0
        articlesStored = 0

        // Containers for data about current article
        var pmid = 0
        var year: Int? = null
        var month = 1
        var day = 1
        var title = ""
        var abstractText = ""

        val authors = mutableListOf<Author>()
        var authorName = ""
        val authorAffiliations = mutableListOf<String>()

        val databanks = mutableListOf<DatabankEntry>()
        var databankName = ""
        val databankAccessionNumbers = mutableListOf<String>()

        val keywordList = mutableListOf<String>()
        val citationList = mutableListOf<Int>()
        val meshHeadingList = mutableListOf<String>()
        var currentMeshHeading = ""

        var journalName = ""
        var language = ""
        var doi = ""
        var type = PublicationType.Article

        // Auxiliary variables for parsing
        var fullName = ""
        var isAbstractStructured = false
        var isArticleTitleParsed = false
        var isAbstractTextParsed = false
        var isCitationPMIDFound = false
        var isDOIFound = false

        // Reset after previously processed files
        articleList.clear()
        deletedArticlePMIDList.clear()

        while (eventReader.hasNext()) {
            val xmlEvent = eventReader.nextEvent()

            // <StartOfTheTag>
            if (xmlEvent.isStartElement) {
                val startElement = xmlEvent.asStartElement()

                // Update full name of the tag -- new tag started
                val localName = startElement.name.localPart
                fullName = if (fullName.isEmpty()) localName else "$fullName/$localName"

                // Process start elements
                when (fullName) {
                    ARTICLE_TAG -> {
                        logger.debug("New article start")
                        isAbstractStructured = false
                        isArticleTitleParsed = false
                        isAbstractTextParsed = false
                        isCitationPMIDFound = false
                        isDOIFound = false

                        pmid = 0
                        year = null
                        month = 1
                        day = 1
                        title = ""
                        abstractText = ""

                        authors.clear()
                        authorName = ""
                        authorAffiliations.clear()

                        databanks.clear()
                        databankName = ""
                        databankAccessionNumbers.clear()

                        keywordList.clear()
                        citationList.clear()
                        meshHeadingList.clear()

                        type = PublicationType.Article
                        journalName = ""
                        language = ""
                        doi = ""
                    }
                    AUTHOR_TAG -> {
                        authorName = ""
                        authorAffiliations.clear()
                    }
                    ABSTRACT_TAG -> {
                        if (startElement.attributes.hasNext()) {
                            isAbstractStructured = true
                        }
                        isAbstractTextParsed = true
                    }

                    // Citations (PMID)
                    CITATION_PMID_TAG -> {
                        if (startElement.getAttributeByName(QName("IdType")).value == "pubmed") {
                            isCitationPMIDFound = true
                        }
                    }

                    // Databanks
                    DATABANK_TAG -> {
                        databankName = ""
                        databankAccessionNumbers.clear()
                    }

                    // DOI
                    DOI_TAG -> {
                        if (startElement.getAttributeByName(QName("IdType")).value == "doi") {
                            isDOIFound = true
                        }
                    }

                    // MeSH
                    MESH_HEADING_TAG -> {
                        currentMeshHeading = ""
                    }
                }

                tags[fullName] = (tags[fullName] ?: 0) + 1
            }

            if (xmlEvent.isCharacters) {
                val dataElement = xmlEvent.asCharacters()

                // Fill the contents of the PubmedArticle class with useful information
                when {
                    // PMID
                    fullName == PMID_TAG -> {
                        pmid = dataElement.data.toInt()
                    }

                    // PMIDs of deleted articles
                    fullName == DELETED_PMID_TAG -> {
                        deletedArticlePMIDList.add(dataElement.data.toInt())
                    }

                    // Year of publication
                    fullName == YEAR_TAG -> {
                        year = dataElement.data.toInt()
                    }
                    fullName == MONTH_TAG -> {
                        month = CALENDAR_MONTH[dataElement.data] ?: 1
                    }
                    fullName == DAY_TAG -> {
                        day = dataElement.data.toInt()
                    }
                    fullName == MEDLINE_TAG -> {
                        val yearRegex = "(1\\d|20)\\d{2}".toRegex()
                        val yearMatch = yearRegex.find(dataElement.data)

                        if (yearMatch != null) {
                            year = yearMatch.value.toInt()
                        } else {
                            logger.warn(
                                    "Failed to parse year from MEDLINE date in article $pmid: " +
                                            dataElement.data
                            )
                        }

                        val monthRegex = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)".toRegex()
                        val monthMatch = monthRegex.find(dataElement.data)

                        if (monthMatch != null) {
                            if (monthMatch.value in CALENDAR_MONTH.keys) {
                                month = CALENDAR_MONTH[monthMatch.value] ?: 1
                            } else {
                                logger.debug(
                                        "Failed to parse name of MEDLINE month in article $pmid: " +
                                                dataElement.data
                                )
                            }
                        } else {
                            logger.debug(
                                    "Failed to parse value of MEDLINE month in article $pmid: " +
                                            dataElement.data
                            )
                        }
                    }

                    // Title
                    isArticleTitleParsed -> {
                        title += dataElement.data
                    }
                    fullName == TITLE_TAG -> {
                        title = dataElement.data
                        isArticleTitleParsed = true
                    }

                    // Abstract
                    isAbstractTextParsed -> {
                        abstractText += if (fullName == ABSTRACT_TAG)
                            dataElement.data
                        else dataElement.data.trim { it <= ' ' }
                    }
                    fullName == OTHER_ABSTRACT_TAG -> {
                        abstractText += " ${dataElement.data}"
                    }

                    // Keywords
                    fullName == KEYWORD_TAG -> {
                        keywordList.add(dataElement.data.trim().replace(", ", " ").replace('/', ' '))
                    }

                    // Citations
                    (fullName == CITATION_PMID_TAG) && (isCitationPMIDFound) -> {
                        citationList.add(dataElement.data.toInt())
                        isCitationPMIDFound = false
                    }

                    // Databanks
                    fullName == DATABANK_NAME_TAG -> {
                        databankName = dataElement.data
                    }
                    fullName == ACCESSION_NUMBER_TAG -> {
                        databankAccessionNumbers.add(dataElement.data)
                    }

                    // MeSH
                    fullName == MESH_DESCRIPTOR_TAG -> {
                        val descriptor = dataElement.data.replace(", ", " ").replace("[/&]+", " ").trim()
                        currentMeshHeading = descriptor
                        logger.debug("$pmid: MeSH Descriptor <$descriptor> <$currentMeshHeading>")
                    }
                    fullName == MESH_QUALIFIER_TAG -> {
                        val qualifier = dataElement.data.replace(", ", " ").replace("[/&]+", " ").trim()
                        currentMeshHeading += " $qualifier"
                        logger.debug("$pmid: MeSH Qualifier <$qualifier> <$currentMeshHeading>")
                    }

                    // Authors
                    fullName == AUTHOR_LASTNAME_TAG -> {
                        authorName = dataElement.data
                    }
                    fullName == AUTHOR_INITIALS_TAG -> {
                        authorName += " ${dataElement.data}"
                    }
                    fullName == AUTHOR_AFFILIATION_TAG -> {
                        authorAffiliations.add(dataElement.data.trim(' ', '.'))
                    }

                    // Other information - journal title, language, DOI, publication type
                    fullName == JOURNAL_TITLE_TAG -> {
                        journalName = dataElement.data
                    }
                    fullName == LANGUAGE_TAG -> {
                        language = dataElement.data
                    }
                    (fullName == DOI_TAG) && (isDOIFound) -> {
                        doi = dataElement.data
                        isDOIFound = false
                    }
                    fullName == PUBLICATION_TYPE_TAG -> {
                        val publicationType = dataElement.data
                        when {
                            publicationType.contains("Technical Report") -> {
                                type = PublicationType.TechnicalReport
                            }
                            publicationType.contains("Clinical Trial") -> {
                                type = PublicationType.ClinicalTrial
                            }
                            publicationType == "Dataset" -> {
                                type = PublicationType.Dataset
                            }
                            publicationType == "Review" -> {
                                type = PublicationType.Review
                            }
                        }
                    }
                }
            }

            // </EndOfTheTag>
            if (xmlEvent.isEndElement) {
                val endElement = xmlEvent.asEndElement()
                val localName = endElement.name.localPart

                when (fullName) {
                    // Add article to the list and store if needed
                    ARTICLE_TAG -> {
                        articleCounter++
                        citationCounter += citationList.size
                        keywordCounter += keywordList.size

                        // There is an issue when some local dates might be absent in other calendars
                        val date = try {
                            if (year != null) DateTime(year, month, day, 12, 0) else null
                        } catch (e: IllegalFieldValueException) {
                            if (year != null) DateTime(year, 1, 1, 12, 0) else null
                        }

                        abstractText = abstractText.trim()
                        articleList.add(
                                PubmedArticle(
                                        pmid = pmid,
                                        date = date,
                                        title = title,
                                        abstractText = abstractText,
                                        keywordList = keywordList.toList(),
                                        citationList = citationList.toList(),
                                        meshHeadingList = meshHeadingList.toList(),
                                        type = type,
                                        doi = doi,
                                        auxInfo = ArticleAuxInfo(
                                                authors.toList(), databanks.toList(), Journal(journalName), language
                                        )
                                )
                        )

                        logger.debug("Found new article")
                        articleList.last().description().forEach {
                            logger.debug("${it.key}: ${it.value}")
                        }
                    }

                    // Add author to the list of authors
                    AUTHOR_TAG -> {
                        authors.add(
                                Author(
                                        authorName, authorAffiliations.toList()
                                )
                        )
                    }

                    // Fix title & abstract
                    ABSTRACT_TAG -> {
                        if (isAbstractStructured) {
                            abstractText += " "
                        }
                        isAbstractTextParsed = false
                    }

                    // Databanks
                    DATABANK_TAG -> {
                        databanks.add(
                                DatabankEntry(
                                        databankName, databankAccessionNumbers.toList()
                                )
                        )
                    }

                    TITLE_TAG -> {
                        title = title.trim('[', ']', '.')
                        isArticleTitleParsed = false
                    }

                    // MeSH
                    MESH_HEADING_TAG -> {
                        meshHeadingList.add(currentMeshHeading)
                    }
                }

                // Update the full name of the tag -- tag closed
                if (localName != null) {
                    fullName = fullName.removeSuffix("/$localName")
                }
            }

            // Stop parsing if found 'parserLimit' articles
            if ((parserLimit > 0) && (articleCounter == parserLimit)) {
                break
            }

            // Store articles if reached preferred size of the batch
            if ((batchSize > 0) && (articleList.size == batchSize)) {
                storeArticles()
                articleList.clear()
            }
        }

        // Store final batch of articles if not empty
        if (articleList.size > 0) {
            storeArticles()
        }

        // Delete articles if needed
        if (deletedArticlePMIDList.size > 0) {
            logger.info("Deleting ${deletedArticlePMIDList.size} articles")

            dbHandler.delete(deletedArticlePMIDList)
        }

        logger.info(
                "Articles found: $articleCounter, deleted: ${deletedArticlePMIDList.size}, " +
                        "keywords: $keywordCounter, citations: $citationCounter"
        )
    }

    private fun storeArticles() {
        logger.info("Storing articles ${articlesStored + 1}-${articlesStored + articleList.size}...")

        dbHandler.store(articleList)
        articlesStored += articleList.size
    }
}