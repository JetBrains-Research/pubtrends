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

    private fun checkArticles(articles: List<PubmedArticle>, articlesMap: Map<Int, PubmedArticle>) {
        var articlesChecked = 0

        assertEquals(articlesMap.size, articles.size, "Wrong number of articles")
        articles.forEach {
            checkEquality(articlesMap[it.pmid]!!, it)
            articlesChecked++
        }
        assertEquals(articlesChecked, articlesMap.size)
    }

    private fun testArticlesForFile(name : String, articlesMap: Map<Int, PubmedArticle>) {
        val path = parserFileSetup(name)
        parser.parse(path)

        val articles = parser.pubmedXMLHandler.articles
        checkArticles(articles, articlesMap)

        val testFile = File(path)
        testFile.delete()
        assertTrue { !testFile.exists() }
    }

    @Test
    fun testParseFormattedAbstract() {
        val formattedArticles = mapOf(29736257 to Articles.article29736257,
                29456534 to Articles.article29456534,
                20453483 to Articles.article20453483,
                27654823 to Articles.article27654823)
        testArticlesForFile("articlesWithFormattedAbstract.xml", formattedArticles)
    }

    @Test
    fun testParseOtherAbstract() {
        val otherArticles = mapOf(11243089 to Articles.article11243089,
                11540070 to Articles.article11540070)
        testArticlesForFile("articlesWithOtherAbstractField.xml", otherArticles)
    }

    @Test
    fun testParseMedlineDate() {
        val medlineDateArticles = mapOf(10188493 to Articles.article10188493,
                14316043 to Articles.article14316043,
                18122624 to Articles.article18122624)
        testArticlesForFile("articlesWithMedlineDateField.xml", medlineDateArticles)
    }
}