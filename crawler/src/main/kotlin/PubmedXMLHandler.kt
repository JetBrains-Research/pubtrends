import java.lang.Exception

import org.xml.sax.*
import org.xml.sax.helpers.*
import org.apache.logging.log4j.LogManager

import kotlin.collections.HashMap


class PubmedXMLHandler(private val limit : Int = 10) : DefaultHandler() {
    private val logger = LogManager.getLogger(PubmedXMLHandler::class)
    private val tags = HashMap<String, Int>()
    private var articleCounter = 0
    private var citationCounter = 0
    private var keywordCounter = 0
    private var skip = false
    private var fullName = String()
    private var currentArticle = PubmedArticle(0)

    val articles : MutableList<PubmedArticle> = mutableListOf()

    companion object {
        const val articleTag = "PubmedArticleSet/PubmedArticle"
        private const val medlineCitationTag = "PubmedArticleSet/PubmedArticle/MedlineCitation"

        const val abstractTag = "$medlineCitationTag/Article/Abstract/AbstractText"
        const val otherAbstractTag = "$medlineCitationTag/OtherAbstract/AbstractText"
        const val citationTag = "$medlineCitationTag/CommentsCorrectionsList/CommentsCorrections"
        const val citationIDTag = "$citationTag/PMID"
        const val keywordTag = "$medlineCitationTag/KeywordList/Keyword"
        const val medlineTag = "$medlineCitationTag/Article/Journal/JournalIssue/PubDate/MedlineDate"
        const val pmidTag = "$medlineCitationTag/PMID"
        const val titleTag = "$medlineCitationTag/Article/ArticleTitle"
        const val yearTag = "$medlineCitationTag/Article/Journal/JournalIssue/PubDate/Year"
    }

    private var abstractState = false
    private var keywordState = false
    private var citationState = false
    private var citationIDState = false
    private var medlineState = false
    private var pmidState = false
    private var titleState = false
    private var yearState = false

    override fun startDocument() {
        articleCounter = 0
        articles.clear()

        citationCounter = 0
        keywordCounter = 0

        abstractState = false
        keywordState = false
        citationState = false
        citationIDState = false
        medlineState = false
        pmidState = false
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
                if (fullName.equals(articleTag, ignoreCase = true)) {
                    currentArticle = PubmedArticle(0)
                }
                if ((fullName.equals(abstractTag, ignoreCase = true)) ||
                        (fullName.equals(otherAbstractTag, ignoreCase = true))) {
                    abstractState = true
                }
                if ((fullName.equals(citationTag, ignoreCase = true)) &&
                        (attributes?.getValue("RefType") == "Cites")) {
                    citationState = true
                }
                if ((citationState) && (fullName.equals(citationIDTag, ignoreCase = true))) {
                    citationIDState = true
                }
                if (fullName.equals(keywordTag, ignoreCase = true)) {
                    keywordState = true
                }
                if (fullName.equals(medlineTag, ignoreCase = true)) {
                    medlineState = true
                }
                if (fullName.equals(pmidTag, ignoreCase = true)) {
                    pmidState = true
                }
                if (fullName.equals(titleTag, ignoreCase = true)) {
                    titleState = true
                }
                if (fullName.equals(yearTag, ignoreCase = true)) {
                    yearState = true
                }
                tags[fullName] = (tags[fullName] ?: 0) + 1
            }
        }
    }

    override fun endElement(uri: String?, localName: String?, qName: String?) {
        if (!skip) {
            if (fullName.equals(articleTag, ignoreCase = true)) {
                articleCounter++
                citationCounter += currentArticle.citationList.size
                keywordCounter += currentArticle.keywordList.size
                articles.add(currentArticle)

                logger.debug("Article ${currentArticle.pmid}")
                currentArticle.description().forEach {
                    logger.debug("${it.key}: ${it.value}")
                }

                if ((limit > 0) && (articleCounter == limit)) {
                    skip = true
                }
            }
            if (localName != null) {
                fullName = fullName.removeSuffix("/$localName")
            }
        }
    }

    override fun characters(ch: CharArray?, start: Int, length: Int) {
        if (!skip) {
            val data = if (ch != null) String(ch, start, length) else ""

            if (abstractState) {
                currentArticle.abstractText += data
                abstractState = false
            }
            if (citationIDState) {
                currentArticle.citationList.add(data.toInt())
                citationState = false
                citationIDState = false
            }
            if (keywordState) {
                currentArticle.keywordList.add(data)
                keywordState = false
            }
            if (medlineState) {
                val regex = "^(19|20)\\d{2}\$".toRegex()
                val match = regex.find(data)
                try {
                    currentArticle.year = match?.value?.toInt()
                } catch (e : Exception) {
                    logger.warn("Failed to parse MEDLINE date: $data")
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