package org.jetbrains.bio.pubtrends.biorxiv

import org.apache.logging.log4j.LogManager
import org.joda.time.format.DateTimeFormat
import org.jsoup.Jsoup

class BiorxivScraper {
    companion object {
        private val logger = LogManager.getLogger(BiorxivScraper::class)
        private val formatter = DateTimeFormat.forPattern("yyyy-mm-dd")

        const val BIORXIV_BASE_URL = "http://biorxiv.org"
        const val BIORXIV_LIST_URL = "http://biorxiv.org/content/early/recent"

        fun getArticleListPageURL(page: Int): String = "$BIORXIV_LIST_URL?page=$page"
        fun getArticlePageURL(link: String): String = "$BIORXIV_BASE_URL$link"
    }

    val pagesCount = Jsoup.connect(BIORXIV_LIST_URL).get().select("li.pager-last > a").text().toInt()

    fun extractArticleLinks(amount: Int = pagesCount): List<String> {
        val articleLinks = mutableListOf<String>()
        for (i in 0 until amount) {
            val articlesListPage = Jsoup.connect(getArticleListPageURL(i)).get()
            val links = articlesListPage.select("span.highwire-cite-title > a[href]").map {
                it.attr("href")
            }

            logger.info("Page ${i + 1} / $amount")
            articleLinks.addAll(links)
        }

        return articleLinks
    }

    fun extractArticle(link: String) : BiorxivArticle {
        val articlePage = Jsoup.connect(getArticlePageURL(link)).get()

        val citationId = articlePage.select("meta[name='citation_id']").attr("content")
        val biorxivId = citationId.split('v')[0]
        val version = citationId.split('v')[1].toInt()

        val title = articlePage.select("meta[name='DC.Title']").attr("content")
        val abstract = articlePage.select("meta[name='DC.Description']").attr("content")
        val dateString = articlePage.select("meta[name='citation_date']").attr("content")
        val date = formatter.parseLocalDate(dateString)

        val authors = articlePage.select("meta[name='DC.Contributor']").map {
            it.attr("content")
        }

        val doi = articlePage.select("meta[name='DC.Identifier']").attr("content")
        val pdfUrl = articlePage.select("meta[name='citation_pdf_url']").attr("content")

        return BiorxivArticle(biorxivId, version, date, title, abstract, authors, doi, pdfUrl)
    }
}
