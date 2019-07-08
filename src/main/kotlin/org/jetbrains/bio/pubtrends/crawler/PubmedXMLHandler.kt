package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.xml.sax.Attributes
import org.xml.sax.helpers.DefaultHandler


class PubmedXMLHandler(
        private val dbHandler: AbstractDBHandler,
        private val parserLimit: Int,
        private val batchSize: Int
) : DefaultHandler() {
    private val logger = LogManager.getLogger(PubmedXMLHandler::class)

    val tags = HashMap<String, Int>()
    private var articleCounter = 0
    private var articlesStored = 0
    private var citationCounter = 0
    private var keywordCounter = 0
    private var skip = false
    private var fullName = ""
    private var currentArticle = PubmedArticle(0)

    private var lastName = ""
    private var initials = ""
    private var affiliation = mutableListOf<String>()

    val articles: MutableList<PubmedArticle> = mutableListOf()

    companion object {
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
        const val TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/ArticleTitle"
        const val YEAR_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/Year"
    }

    private var abstractState = false
    private var authorLastNameState = false
    private var authorInitialsState = false
    private var authorAffiliationState = false
    private var doiState = false
    private var keywordState = false
    private var languageState = false
    private var journalTitleState = false
    private var citationIDState = false
    private var medlineState = false
    private var pmidState = false
    private var publicationTypeState = false
    private var structuredAbstract = false
    private var titleState = false
    private var yearState = false

    override fun startDocument() {
        articleCounter = 0
        articlesStored = 0
        articles.clear()

        citationCounter = 0
        keywordCounter = 0

        abstractState = false
        authorLastNameState = false
        authorInitialsState = false
        authorAffiliationState = false
        keywordState = false
        languageState = false
        citationIDState = false
        journalTitleState = false
        medlineState = false
        pmidState = false
        publicationTypeState = false
        structuredAbstract = false
        titleState = false
        yearState = false

        skip = false

        fullName = ""
        currentArticle = PubmedArticle(0)
    }

    override fun endDocument() {
        if (articles.size > 0) {
            storeArticles()
        }
        logger.info("Articles found: $articleCounter, stored: $articlesStored, " +
                "keywords: $keywordCounter, citations: $citationCounter")
    }

    override fun startElement(uri: String?, localName: String?, qName: String?, attributes: Attributes?) {
        if (!skip) {
            if (localName != null) {
                fullName = if (fullName.isEmpty()) localName else "$fullName/$localName"
                when {
                    fullName.equals(ARTICLE_TAG, ignoreCase = true) -> {
                        currentArticle = PubmedArticle(0)
                    }
                    fullName.equals(AUTHOR_LASTNAME_TAG, ignoreCase = true) -> {
                        authorLastNameState = true
                    }
                    fullName.equals(AUTHOR_INITIALS_TAG, ignoreCase = true) -> {
                        authorInitialsState = true
                    }
                    fullName.equals(AUTHOR_AFFILIATION_TAG, ignoreCase = true) -> {
                        authorAffiliationState = true
                    }
                    fullName.equals(ABSTRACT_TAG, ignoreCase = true) -> {
                        if ((attributes != null) && (attributes.length > 0)) {
                            structuredAbstract = true
                        }
                        abstractState = true
                    }
                    fullName.equals(DOI_TAG, ignoreCase = true) -> {
                        if ((attributes != null) && (attributes.getValue("ValidYN") == "Y")) {
                            doiState = true
                        }
                    }
                    fullName.equals(JOURNAL_TITLE_TAG, ignoreCase = true) -> {
                        journalTitleState = true
                    }
                    fullName.equals(OTHER_ABSTRACT_TAG, ignoreCase = true) -> {
                        abstractState = true
                        currentArticle.abstractText += " "
                    }
                    fullName.equals(CITATION_PMID_TAG, ignoreCase = true) -> {
                        if ((attributes != null) && (attributes.getValue("IdType") == "pubmed")) {
                            citationIDState = true
                        }
                    }
                    fullName.equals(LANGUAGE_TAG, ignoreCase = true) -> {
                        languageState = true
                    }
                    fullName.equals(KEYWORD_TAG, ignoreCase = true) -> {
                        keywordState = true
                    }
                    fullName.equals(MEDLINE_TAG, ignoreCase = true) -> {
                        medlineState = true
                    }
                    fullName.equals(PMID_TAG, ignoreCase = true) -> {
                        pmidState = true
                    }
                    fullName.equals(TITLE_TAG, ignoreCase = true) -> {
                        titleState = true
                    }
                    fullName.equals(YEAR_TAG, ignoreCase = true) -> {
                        yearState = true
                    }
                }
                tags[fullName] = (tags[fullName] ?: 0) + 1
            }
        }
    }

    override fun endElement(uri: String?, localName: String?, qName: String?) {
        if (!skip) {
            when {
                fullName.equals(ARTICLE_TAG, ignoreCase = true) -> {
                    articleCounter++
                    citationCounter += currentArticle.citationList.size
                    keywordCounter += currentArticle.keywordList.size
                    currentArticle.title = currentArticle.title.trim()
                    currentArticle.abstractText = currentArticle.abstractText.trim()

                    articles.add(currentArticle)

                    logger.debug("Article ${currentArticle.pmid}")
                    currentArticle.description().forEach {
                        logger.debug("${it.key}: ${it.value}")
                    }

                    if ((parserLimit > 0) && (articleCounter == parserLimit)) {
                        skip = true
                    }
                }
                (fullName.equals(ABSTRACT_TAG, ignoreCase = true)) ||
                        (fullName.equals(OTHER_ABSTRACT_TAG, ignoreCase = true)) -> {
                    if (structuredAbstract) {
                        currentArticle.abstractText += " "
                    }
                    abstractState = false
                }
                fullName.equals(AUTHOR_TAG, ignoreCase = true) -> {
                    currentArticle.auxInfo.authors.add(Author("$lastName $initials"))

                    lastName = ""
                    initials = ""
                    affiliation.clear()
                }
                fullName.equals(TITLE_TAG, ignoreCase = true) -> {
                    currentArticle.title = currentArticle.title.trim('[', ']', '.')
                    titleState = false
                }
            }
            if (localName != null) {
                fullName = fullName.removeSuffix("/$localName")
            }
        }

        if (articles.size == batchSize) {
            storeArticles()
            articles.clear()
        }
    }

    override fun characters(ch: CharArray?, start: Int, length: Int) {
        if (!skip) {
            val data = if (ch != null) String(ch, start, length) else ""

            if (abstractState) {
                if (!fullName.equals(ABSTRACT_TAG, ignoreCase = true)) {
                    currentArticle.abstractText += data.trim { it <= ' ' }
                } else {
                    currentArticle.abstractText += data
                }
            }
            if (authorLastNameState) {
                lastName = data
                authorLastNameState = false
            }
            if (authorInitialsState) {
                initials = data
                authorInitialsState = false
            }
            if (authorAffiliationState) {
                affiliation.add(data)
                authorAffiliationState = false
            }
            if (doiState) {
                currentArticle.doi = data.trim { it <= ' ' }
                doiState = false
            }
            if (journalTitleState)
            {
                currentArticle.auxInfo.journal.name = data
                journalTitleState = false
            }
            if (citationIDState) {
                currentArticle.citationList.add(data.toInt())
                citationIDState = false
            }
            if (keywordState) {
                currentArticle.keywordList.add(data)
                keywordState = false
            }
            if (languageState) {
                currentArticle.auxInfo.language = data
            }
            if (medlineState) {
                val regex = "(19|20)\\d{2}".toRegex()
                val match = regex.find(data)
                try {
                    currentArticle.year = match?.value?.toInt()
                } catch (e: Exception) {
                    logger.warn("Failed to parse MEDLINE date in article ${currentArticle.pmid}: $data")
                }
                medlineState = false
            }
            if (pmidState) {
                currentArticle.pmid = data.toInt()
                pmidState = false
            }
            if (titleState) {
                if (!fullName.equals(TITLE_TAG, ignoreCase = true)) {
                    currentArticle.title += data.trim { it <= ' ' }
                } else {
                    currentArticle.title += data
                }
            }
            if (yearState) {
                currentArticle.year = data.toInt()
                yearState = false
            }
        }
    }

    private fun storeArticles() {
        logger.info("Storing articles ${articlesStored + 1}-${articlesStored + articles.size}...")

        dbHandler.store(articles)
        articlesStored += articles.size
    }
}