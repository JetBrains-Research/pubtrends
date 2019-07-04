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
    private var fullName = String()
    private var currentArticle = PubmedArticle(0)

    val articles: MutableList<PubmedArticle> = mutableListOf()

    companion object {
        const val ARTICLE_TAG = "PubmedArticleSet/PubmedArticle"
        private const val MEDLINE_CITATION_TAG = "PubmedArticleSet/PubmedArticle/MedlineCitation"
        private const val PUBMED_DATA_TAG = "PubmedArticleSet/PubmedArticle/PubmedData"

        const val ABSTRACT_TAG = "$MEDLINE_CITATION_TAG/Article/Abstract/AbstractText"
        const val OTHER_ABSTRACT_TAG = "$MEDLINE_CITATION_TAG/OtherAbstract/AbstractText"
        const val CITATION_PMID_TAG = "$PUBMED_DATA_TAG/ReferenceList/Reference/ArticleIdList/ArticleId"
        const val KEYWORD_TAG = "$MEDLINE_CITATION_TAG/KeywordList/Keyword"
        const val MEDLINE_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/MedlineDate"
        const val PMID_TAG = "$MEDLINE_CITATION_TAG/PMID"
        const val TITLE_TAG = "$MEDLINE_CITATION_TAG/Article/ArticleTitle"
        const val YEAR_TAG = "$MEDLINE_CITATION_TAG/Article/Journal/JournalIssue/PubDate/Year"
    }

    private var abstractState = false
    private var keywordState = false
    private var citationIDState = false
    private var hasOtherVersion = false
    private var medlineState = false
    private var pmidState = false
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
        keywordState = false
        citationIDState = false
        medlineState = false
        pmidState = false
        structuredAbstract = false
        titleState = false
        yearState = false

        skip = false

        fullName = String()
        currentArticle = PubmedArticle(0)
    }

    override fun endDocument() {
        logger.info("Articles: $articleCounter, keywords: $keywordCounter, citations: $citationCounter")
    }

    override fun startElement(uri: String?, localName: String?, qName: String?, attributes: Attributes?) {
        if (!skip) {
            if (localName != null) {
                fullName = if (fullName.isEmpty()) localName else "$fullName/$localName"
                if (fullName.equals(ARTICLE_TAG, ignoreCase = true)) {
                    currentArticle = PubmedArticle(0)
                    hasOtherVersion = false
                }
                if (fullName.equals(ABSTRACT_TAG, ignoreCase = true)) {
                    if ((attributes != null) && (attributes.length > 0)) {
                        structuredAbstract = true
                    }
                    abstractState = true
                }
                if (fullName.equals(OTHER_ABSTRACT_TAG, ignoreCase = true)) {
                    abstractState = true
                    currentArticle.abstractText += " "
                }
                if (fullName.equals(CITATION_PMID_TAG, ignoreCase = true) &&
                        ((attributes != null) && (attributes.getValue("IdType") == "pubmed"))) {
                    citationIDState = true
                }
                if (fullName.equals(KEYWORD_TAG, ignoreCase = true)) {
                    keywordState = true
                }
                if (fullName.equals(MEDLINE_TAG, ignoreCase = true)) {
                    medlineState = true
                }
                if (fullName.equals(PMID_TAG, ignoreCase = true)) {
                    hasOtherVersion = (attributes?.getValue("Version") != "1")
                    pmidState = true
                }
                if (fullName.equals(TITLE_TAG, ignoreCase = true)) {
                    titleState = true
                }
                if (fullName.equals(YEAR_TAG, ignoreCase = true)) {
                    yearState = true
                }
                tags[fullName] = (tags[fullName] ?: 0) + 1
            }
        }
    }

    override fun endElement(uri: String?, localName: String?, qName: String?) {
        if (!skip) {
            if (fullName.equals(ARTICLE_TAG, ignoreCase = true)) {
                articleCounter++
                citationCounter += currentArticle.citationList.size
                keywordCounter += currentArticle.keywordList.size
                currentArticle.title = currentArticle.title.trim()
                currentArticle.abstractText = currentArticle.abstractText.trim()
                if (!hasOtherVersion) {
                    articles.add(currentArticle)

                    logger.debug("Article ${currentArticle.pmid}")
                    currentArticle.description().forEach {
                        logger.debug("${it.key}: ${it.value}")
                    }
                }

                if ((parserLimit > 0) && (articleCounter == parserLimit)) {
                    skip = true
                }
            }
            if ((fullName.equals(ABSTRACT_TAG, ignoreCase = true)) ||
                    (fullName.equals(OTHER_ABSTRACT_TAG, ignoreCase = true))) {
                if (structuredAbstract) {
                    currentArticle.abstractText += " "
                }
                abstractState = false
            }
            if (localName != null) {
                fullName = fullName.removeSuffix("/$localName")
            }
        }

        if (articles.size == batchSize) {
            logger.info("Storing articles ${articlesStored + 1}-${articlesStored + articles.size}...")

            dbHandler.store(articles)
            articlesStored += articles.size
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
            if (citationIDState) {
                currentArticle.citationList.add(data.toInt())
                citationIDState = false
            }
            if (keywordState) {
                currentArticle.keywordList.add(data)
                keywordState = false
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
                currentArticle.title += data
                titleState = false
            }
            if (yearState) {
                currentArticle.year = data.toInt()
                yearState = false
            }
        }
    }
}