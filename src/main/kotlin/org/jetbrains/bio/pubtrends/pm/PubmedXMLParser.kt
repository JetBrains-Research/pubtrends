package org.jetbrains.bio.pubtrends.pm

import org.apache.logging.log4j.LogManager
import org.xml.sax.SAXException
import java.io.File
import javax.xml.namespace.QName
import javax.xml.stream.XMLEventReader
import javax.xml.stream.XMLInputFactory


class PubmedXMLParser(
        private val dbHandler: AbstractDBHandler,
        private val parserLimit: Int,
        private val batchSize: Int
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
        const val ACCESSION_NUMBER_TAG = "$DATABANK_TAG/AccessionNumber"

        const val DOI_TAG = "$PUBMED_DATA_TAG/ArticleIdList/ArticleId"
        const val LANGUAGE_TAG = "$MEDLINE_CITATION_TAG/Article/Language"
        const val JOURNAL_TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/Title"
    }

    private val factory = XMLInputFactory.newFactory()!!

    // Container for parsed articles
    internal val articleList = mutableListOf<PubmedArticle>()
    internal val deletedArticlePMIDList = mutableListOf<Int>()

    // Stats about articles & XML tags
    val tags = HashMap<String, Int>()
    private var articleCounter = 0
    private var articlesStored = 0
    private var citationCounter = 0
    private var keywordCounter = 0

    // Temporary containers for information about articles and authors respectively
    private var currentArticle = PubmedArticle()
    private var currentAuthor = Author()
    private var currentMeshHeading : String = ""
    private var currentDatabankEntry = DatabankEntry()

    // Auxiliary variables for parsing
    private var fullName = ""
    private var isAbstractStructured = false
    private var isArticleTitleParsed = false
    private var isAbstractTextParsed = false
    private var isCitationPMIDFound = false
    private var isDOIFound = false

    fun parse(name: String): Boolean {
        try {
            logger.debug("File location: ${File(name).absolutePath}")
            File(name).inputStream().use {
                parseData(factory.createXMLEventReader(it))
            }
        } catch (e: SAXException) {
            logger.error("Failed to parse $name", e)
        }

        return true
    }

    fun parseData(eventReader : XMLEventReader) {
        articleCounter = 0
        articlesStored = 0
        articleList.clear()
        deletedArticlePMIDList.clear()
        keywordCounter = 0
        citationCounter = 0
        tags.clear()
        fullName = ""

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
                        currentArticle = PubmedArticle()
                        isAbstractStructured = false
                        isArticleTitleParsed = false
                        isAbstractTextParsed = false
                        isCitationPMIDFound = false
                        isDOIFound = false
                    }
                    AUTHOR_TAG -> {
                        currentAuthor = Author()
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
                        currentDatabankEntry = DatabankEntry()
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
                        currentArticle.pmid = dataElement.data.toInt()
                    }

                    // PMIDs of deleted articles
                    fullName == DELETED_PMID_TAG -> {
                        deletedArticlePMIDList.add(dataElement.data.toInt())
                    }

                    // Year of publication
                    fullName == YEAR_TAG -> {
                        currentArticle.year = dataElement.data.toInt()
                    }
                    fullName == MEDLINE_TAG -> {
                        val regex = "(19|20)\\d{2}".toRegex()
                        val match = regex.find(dataElement.data)
                        try {
                            currentArticle.year = match?.value?.toInt()
                        } catch (e: Exception) {
//                        logger.warn("Failed to parse MEDLINE date in article ${currentArticle.pmid}: $data")
                        }
                    }

                    // Title
                    isArticleTitleParsed -> {
                        currentArticle.title += dataElement.data
                    }
                    fullName == TITLE_TAG -> {
                        currentArticle.title = dataElement.data
                        isArticleTitleParsed = true
                    }

                    // Abstract
                    isAbstractTextParsed -> {
                        if (fullName == ABSTRACT_TAG) {
                            currentArticle.abstractText += dataElement.data
                        } else {
                            currentArticle.abstractText += dataElement.data.trim { it <= ' ' }
                        }
                    }
                    fullName == OTHER_ABSTRACT_TAG -> {
                        currentArticle.abstractText += " ${dataElement.data}"
                    }

                    // Keywords
                    fullName == KEYWORD_TAG -> {
                        currentArticle.keywordList.add(dataElement.data)
                    }

                    // Citations
                    fullName == CITATION_PMID_TAG -> {
                        currentArticle.citationList.add(dataElement.data.toInt())
                        isCitationPMIDFound = false
                    }

                    // Databanks
                    fullName == DATABANK_NAME_TAG -> {
                        currentDatabankEntry.name = dataElement.data
                    }
                    fullName == ACCESSION_NUMBER_TAG -> {
                        currentDatabankEntry.accessionNumber.add(dataElement.data)
                    }

                    // MeSH
                    fullName == MESH_DESCRIPTOR_TAG -> {
                        currentMeshHeading = dataElement.data
                    }
                    fullName == MESH_QUALIFIER_TAG -> {
                        currentMeshHeading += " ${dataElement.data}"
                    }

                    // Authors
                    fullName == AUTHOR_LASTNAME_TAG -> {
                        currentAuthor.name = dataElement.data
                    }
                    fullName == AUTHOR_INITIALS_TAG -> {
                        currentAuthor.name += " ${dataElement.data}"
                    }
                    fullName == AUTHOR_AFFILIATION_TAG -> {
                        currentAuthor.affiliation.add(dataElement.data)
                    }

                    // Other information - journal title, language, DOI, publication type
                    fullName == JOURNAL_TITLE_TAG -> {
                        currentArticle.auxInfo.journal.name = dataElement.data
                    }
                    fullName == LANGUAGE_TAG -> {
                        currentArticle.auxInfo.language = dataElement.data
                    }
                    fullName == DOI_TAG -> {
                        currentArticle.doi = dataElement.data
                        isDOIFound = false
                    }
                    fullName == PUBLICATION_TYPE_TAG -> {
                        val publicationType = dataElement.data
                        when {
                            publicationType.contains("Technical Report") -> {
                                currentArticle.type = PublicationType.TechnicalReport
                            }
                            publicationType.contains("Clinical Trial") -> {
                                currentArticle.type = PublicationType.ClinicalTrial
                            }
                            publicationType.equals("Dataset") -> {
                                currentArticle.type = PublicationType.Dataset
                            }
                            publicationType.equals("Review") -> {
                                currentArticle.type = PublicationType.Review
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
                        citationCounter += currentArticle.citationList.size
                        keywordCounter += currentArticle.keywordList.size

                        currentArticle.abstractText = currentArticle.abstractText.trim()
                        articleList.add(currentArticle)

                        logger.debug("Found new article")
                        currentArticle.description().forEach {
                            logger.debug("${it.key}: ${it.value}")
                        }
                    }

                    // Add author to the list of authors
                    AUTHOR_TAG -> {
                        currentArticle.auxInfo.authors.add(currentAuthor)
                    }

                    // Fix title & abstract
                    ABSTRACT_TAG -> {
                        if (isAbstractStructured) {
                            currentArticle.abstractText += " "
                        }
                        isAbstractTextParsed = false
                    }

                    // Databanks
                    DATABANK_TAG -> {
                        currentArticle.databankEntryList.add(currentDatabankEntry)
                    }

                    TITLE_TAG -> {
                        currentArticle.title = currentArticle.title.trim('[', ']', '.')
                        isArticleTitleParsed = false
                    }

                    // MeSH
                    MESH_HEADING_TAG -> {
                        currentArticle.meshHeadingList.add(currentMeshHeading)
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
            if (articleList.size == batchSize) {
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

        logger.info("Articles found: $articleCounter, deleted: ${deletedArticlePMIDList.size}, " +
                "keywords: $keywordCounter, citations: $citationCounter")
    }

    private fun storeArticles() {
        logger.info("Storing articles ${articlesStored + 1}-${articlesStored + articleList.size}...")

        dbHandler.store(articleList)
        articlesStored += articleList.size
    }
}