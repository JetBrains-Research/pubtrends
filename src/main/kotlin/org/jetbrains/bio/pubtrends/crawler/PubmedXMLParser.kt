package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.xml.sax.SAXException
import java.io.File
import javax.xml.namespace.QName
import javax.xml.stream.XMLEventReader
import javax.xml.stream.XMLInputFactory
import javax.xml.stream.events.Characters


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
        const val ABSTRACT_TAG = "$MEDLINE_CITATION_TAG/Article/Abstract/AbstractText"
        const val AUTHOR_TAG = "$MEDLINE_CITATION_TAG/Article/AuthorList/Author"
        const val AUTHOR_LASTNAME_TAG = "$MEDLINE_CITATION_TAG/Article/AuthorList/Author/LastName"
        const val AUTHOR_INITIALS_TAG = "$MEDLINE_CITATION_TAG/Article/AuthorList/Author/Initials"
        const val AUTHOR_AFFILIATION_TAG = "$MEDLINE_CITATION_TAG/Article/AuthorList/Author/AffiliationInfo/Affiliation"
        const val DOI_TAG = "$PUBMED_DATA_TAG/ArticleIdList/ArticleId"
        const val OTHER_ABSTRACT_TAG = "$MEDLINE_CITATION_TAG/OtherAbstract/AbstractText"
        const val LANGUAGE_TAG = "$MEDLINE_CITATION_TAG/Article/Language"
        const val JOURNAL_TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/Title"
        const val CITATION_PMID_TAG = "$PUBMED_DATA_TAG/ReferenceList/Reference/ArticleIdList/ArticleId"
        const val KEYWORD_TAG = "$MEDLINE_CITATION_TAG/KeywordList/Keyword"
        const val MEDLINE_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/MedlineDate"
        const val PMID_TAG = "$MEDLINE_CITATION_TAG/PMID"
        const val PUBLICATION_TYPE_TAG = "$MEDLINE_CITATION_TAG/Article/PublicationTypeList/PublicationType"
        const val TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/ArticleTitle"
        const val YEAR_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/Year"
    }

    private val factory = XMLInputFactory.newFactory()!!

    // Container for parsed articles
    internal val articleList = mutableListOf<PubmedArticle>()

    // Stats about articles & XML tags
    val tags = HashMap<String, Int>()
    private var articleCounter = 0
    private var articlesStored = 0
    private var citationCounter = 0
    private var keywordCounter = 0

    // Temporary containers for information about articles and authors respectively
    private var currentArticle = PubmedArticle()
    private var currentAuthor = Author()

    // Auxiliary variables for parsing
    private var fullName = ""
    private var isAbstractStructured = false
    private var isArticleTitleParsed = false
    private var isAbstractTextParsed = false

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
                    }
                    AUTHOR_TAG -> {
                        currentAuthor = Author()
                    }
                    ABSTRACT_TAG -> {
                        if (startElement.attributes.hasNext()) {
                            isAbstractStructured = true
                        }
                    }

                    // Citations (PMID)
                    CITATION_PMID_TAG -> {
                        if (startElement.getAttributeByName(QName("IdType")).value == "pubmed") {
                            val citationPMIDDataEvent = eventReader.nextEvent() as Characters
                            currentArticle.citationList.add(citationPMIDDataEvent.data.toInt())
                        }
                    }

                    // DOI
                    DOI_TAG -> {
                        if (startElement.getAttributeByName(QName("IdType")).value == "doi") {
                            val doiDataEvent = eventReader.nextEvent() as Characters
                            currentArticle.doi = doiDataEvent.data
                        }
                    }
                }

                tags[fullName] = (tags[fullName] ?: 0) + 1
            }

            if (xmlEvent.isCharacters) {
                val dataElement = xmlEvent.asCharacters()

                // Fill the contents of the PubmedArticle class with useful information
                when {
                    // PMID
                    fullName.equals(PMID_TAG) -> {
                        currentArticle.pmid = dataElement.data.toInt()
                    }

                    // Year of publication
                    fullName.equals(YEAR_TAG) -> {
                        currentArticle.year = dataElement.data.toInt()
                    }
                    fullName.equals(MEDLINE_TAG) -> {
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
                    fullName.equals(TITLE_TAG) -> {
                        currentArticle.title = dataElement.data
                        isArticleTitleParsed = true
                    }

                    // Abstract
                    isAbstractTextParsed -> {
                        logger.debug("$fullName <${dataElement.data}>")

                        currentArticle.abstractText += dataElement.data.trim {it <= ' '}
                    }
                    fullName.equals(ABSTRACT_TAG) -> {
                        logger.debug("$fullName <${dataElement.data}>")

                        currentArticle.abstractText += dataElement.data

                        isAbstractTextParsed = true
                    }
                    fullName.equals(OTHER_ABSTRACT_TAG) -> {
                        logger.debug("$fullName <${dataElement.data}>")

                        currentArticle.abstractText += " ${dataElement.data}"
                    }

                    // Keywords
                    fullName.equals(KEYWORD_TAG) -> {
                        currentArticle.keywordList.add(dataElement.data)
                    }

                    // Authors
                    fullName.equals(AUTHOR_LASTNAME_TAG) -> {
                        currentAuthor.name = dataElement.data
                    }
                    fullName.equals(AUTHOR_INITIALS_TAG) -> {
                        currentAuthor.name += " ${dataElement.data}"
                    }
                    fullName.equals(AUTHOR_AFFILIATION_TAG) -> {
                        currentAuthor.affiliation.add(dataElement.data)
                    }

                    // Other information - journal title, language, publication type
                    fullName.equals(JOURNAL_TITLE_TAG) -> {
                        currentArticle.auxInfo.journal.name = dataElement.data
                    }
                    fullName.equals(LANGUAGE_TAG) -> {
                        currentArticle.auxInfo.language = dataElement.data
                    }
                    fullName.equals(PUBLICATION_TYPE_TAG) -> {
                        val publicationType = dataElement.data
                        when {
                            publicationType.contains("Technical Report") -> {
                                currentArticle.type = ArticleTypes.TechnicalReport
                            }
                            publicationType.contains("Clinical Trial") -> {
                                currentArticle.type = ArticleTypes.ClinicalTrial
                            }
                            publicationType.equals("Dataset") -> {
                                currentArticle.type = ArticleTypes.Dataset
                            }
                            publicationType.equals("Review") -> {
                                currentArticle.type = ArticleTypes.Review
                            }
                        }
                    }
                }
            }

            // </EndOfTheTag>
            if (xmlEvent.isEndElement) {
                val endElement = xmlEvent.asEndElement()
                val localName = endElement.name.localPart

                // Add article to the list and store if needed
                if (fullName.equals(ARTICLE_TAG)) {
                    articleCounter++
                    citationCounter += currentArticle.citationList.size
                    keywordCounter += currentArticle.keywordList.size

                    articleList.add(currentArticle)

                    logger.debug("Found new article")
                    currentArticle.description().forEach {
                        logger.debug("${it.key}: ${it.value}")
                    }
                }

                // Add author to the list of authors
                if (fullName.equals(AUTHOR_TAG)) {
                    currentArticle.auxInfo.authors.add(currentAuthor)
                }

                // Fix information
                if (fullName.equals(ABSTRACT_TAG)) {
                    if (isAbstractStructured) {
                        currentArticle.abstractText += " "
                    }
                    isAbstractTextParsed = false
                }
                if (fullName.equals(TITLE_TAG)) {
                    currentArticle.title = currentArticle.title.trim('[', ']', '.')
                    isArticleTitleParsed = false
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
        logger.info("Articles found: $articleCounter, stored: $articlesStored, " +
                "keywords: $keywordCounter, citations: $citationCounter")
    }

    private fun storeArticles() {
        logger.info("Storing articles ${articlesStored + 1}-${articlesStored + articleList.size}...")

        dbHandler.store(articleList)
        articlesStored += articleList.size
    }
}