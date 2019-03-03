package org.jetbrains.bio.pubtrends.crawler

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ParserTest {
    companion object {
        private val parser = PubmedXMLParser(MockDBHandler())
    }

    private fun parserFileSetup(name : String) : String {
        this::class.java.classLoader.getResourceAsStream(name).use {
            val file = createTempFile()
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
        parser.parse(path)

        val articles = parser.pubmedXMLHandler.articles
        var articlesChecked = 0

        assertEquals(Articles.size, articles.size, "Wrong number of articles")
        articles.forEach {
            checkEquality(Articles[it.pmid]!!, it)
            articlesChecked++
        }
        assertEquals(articlesChecked, Articles.size)

        val testFile = File(path)
        testFile.delete()
        assertTrue { !testFile.exists() }
    }
}