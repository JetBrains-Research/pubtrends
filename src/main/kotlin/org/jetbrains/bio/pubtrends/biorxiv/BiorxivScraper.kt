package org.jetbrains.bio.pubtrends.biorxiv

import org.apache.logging.log4j.LogManager
import org.joda.time.format.DateTimeFormat
import org.jsoup.Jsoup
import org.jsoup.nodes.Document

class BiorxivScraper {
    companion object {
        private val logger = LogManager.getLogger(BiorxivScraper::class)
        private val formatter = DateTimeFormat.forPattern("yyyy-mm-dd")

        const val BIORXIV_BASE_URL = "http://biorxiv.org"
        const val BIORXIV_LIST_URL = "http://biorxiv.org/content/early/recent"

        fun pageURL(page: Int): String = "$BIORXIV_LIST_URL?page=$page"

        fun articleURL(link: String): String = "$BIORXIV_BASE_URL$link"
    }

    private val pagesCount = Jsoup.connect(BIORXIV_LIST_URL).get().select("li.pager-last > a").text().toInt()

    fun extractArticleLinks(pagesAmount: Int = pagesCount): List<String> {
        val articleLinks = mutableListOf<String>()
        for (i in 0 until pagesAmount) {
            val articlesListPage = Jsoup.connect(pageURL(i)).get()
            val links = articlesListPage.select("span.highwire-cite-title > a[href]").map {
                it.attr("href")
            }

            logger.info("Page ${i + 1} / $pagesAmount")
            articleLinks.addAll(links)
        }

        return articleLinks
    }

    fun extractArticle(link: String) : BiorxivArticle {
        val articlePage = Jsoup.connect(articleURL(link)).get()

        val citationId = articlePage.extractSingleMetaTagContent("citation_id")
        val biorxivId = getId(citationId.split('v')[0])
        val version = citationId.split('v')[1].toInt()

        val title = articlePage.extractSingleMetaTagContent("DC.Title")
        val abstract = articlePage.extractSingleMetaTagContent("DC.Description")
        val dateString = articlePage.extractSingleMetaTagContent("citation_date")
        val date = formatter.parseLocalDate(dateString)

        val authors = articlePage.extractMultipleMetaTagContent("DC.Contributor")

        val doi = articlePage.extractSingleMetaTagContent("DC.Identifier")
        val pdfUrl = articlePage.extractSingleMetaTagContent("citation_pdf_url")

        val references = articlePage.extractMultipleMetaTagContent("citation_reference")

        logger.info("References\n$references")

        return BiorxivArticle(biorxivId, version, date, title, abstract, authors, doi, pdfUrl)
    }

    private fun Document.extractSingleMetaTagContent(tag: String) : String {
        return this.select("meta[name='$tag']").attr("content")
    }

    private fun Document.extractMultipleMetaTagContent(tag: String) : List<String> {
        return this.select("meta[name='$tag']").map {
            it.attr("content")
        }
    }

    /**
     * There are two types of biorxiv IDs at the moment:
     *   - a simple number (e.g., 507244) - old type of IDs
     *   - number with a date in prefix (e.g., 2020.01.15.908376) - new type of IDs
     *
     * This function is used to extract ID in old type.
     *
     * @param biorxivId String containing without version
     * @return ID in old type
     */
    fun getId(biorxivId : String) : Int {
        if (biorxivId.contains('.'))
        {
            return biorxivId.split('.').last().toInt()
        }
        return biorxivId.toInt()
    }
}
