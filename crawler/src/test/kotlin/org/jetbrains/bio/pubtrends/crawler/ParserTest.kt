package org.jetbrains.bio.pubtrends.crawler

import org.junit.AfterClass
import org.junit.Test
import kotlin.test.assertEquals

class ParserTest {
    companion object {
        private val crawler = PubmedCrawler()

        @JvmStatic
        @AfterClass
        fun cleanup() {
            if (crawler.tempDirectory.exists()) {
                crawler.tempDirectory.deleteRecursively()
            }
        }
    }

    private fun parserFileSetup(name : String) : String{
        this::class.java.classLoader.getResourceAsStream(name).use {
            val file = createTempFile(directory = crawler.tempDirectory)
            file.outputStream().use {out ->
                it.copyTo(out)
            }
            return file.absolutePath
        }
    }

    private fun checkEquality(first: PubmedArticle, second: PubmedArticle) {
        assertEquals(first.pmid, second.pmid, "${second.pmid}: Wrong PMID")
        assertEquals(first.year, second.year, "${second.pmid}: Wrong year")
        assertEquals(first.title, second.title, "${second.pmid}: Wrong title")
        assertEquals(first.abstractText, second.abstractText, "${second.pmid}: Wrong abstract")
        assertEquals(first.keywordList, second.keywordList, "${second.pmid}: Wrong keyword list")
        assertEquals(first.citationList, second.citationList, "${second.pmid}: Wrong citation list")
    }

    @Test
    fun testParse() {
        val path = parserFileSetup("articlesWithFormattedAbstract.xml")
        crawler.parse(path)

        val articles = crawler.pubmedXMLHandler.articles
        var articlesChecked = 0

        assertEquals(Articles.size, articles.size, "Wrong number of articles")
        articles.forEach {
            checkEquality(Articles[it.pmid]!!, it)
            articlesChecked++
        }
        assertEquals(articlesChecked, Articles.size)
    }
}