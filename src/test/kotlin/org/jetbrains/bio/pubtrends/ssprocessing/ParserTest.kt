package org.jetbrains.bio.pubtrends.ssprocessing

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals

class ParserTest {
    private fun parserFileSetup(name: String): File {
        this::class.java.classLoader.getResourceAsStream(name).use {
            val file = createTempFile()
            file.outputStream().use { out ->
                it.copyTo(out)
            }
            return file
        }
    }

    private fun checkEquality(first: SemanticScholarArticle, second: SemanticScholarArticle) {
        assertEquals(first.pmid, second.pmid, "${second.pmid}: Wrong PMID")
        assertEquals(first.ssid, second.ssid, "${second.ssid}: Wrong SSID")
        assertEquals(first.title, second.title, "${second.title}: Wrong title")
        assertEquals(first.abstract, second.abstract, "${second.abstract}: Wrong abstract")
        assertEquals(first.year, second.year, "${second.year}: Wrong year")
        assertEquals(first.doi, second.doi, "${second.doi}: Wrong doi")
        assertEquals(first.keywordList, second.keywordList, "${second.keywordList}: Wrong Keyword List")
        assertEquals(first.source, second.source, "${second.source}: Wrong source")
        assertEquals(first.aux, second.aux, "${second.aux}: Wrong aux")
    }

    private fun checkCitations(first: SemanticScholarArticle, second: SemanticScholarArticle) {
        assertEquals(first.citationList, second.citationList, "${second.citationList}: Wrong List Of Citations")

    }

    private fun checkArticles(articles: List<SemanticScholarArticle>, articlesList: List<SemanticScholarArticle>, citations: Boolean = false) {
        var articlesChecked = 0

        assertEquals(articlesList.size, articles.size, "Wrong number of articles")

        for (i in articles.indices) {
            if (citations)
                checkCitations(articlesList[i], articles[i])
            else
                checkEquality(articlesList[i], articles[i])
            articlesChecked++
        }
        assertEquals(articlesChecked, articlesList.size)
    }

    private fun testArticlesForFile(name: String, correctArticles: List<SemanticScholarArticle>) {
        val articlesFile = parserFileSetup(name)

        val parser = ArchiveParser(articlesFile, 1000, addCitations = false, addToDatabase = false)
        parser.parse()
        val parsedArticles = parser.currentArticles
        checkArticles(parsedArticles, correctArticles)

        val citationsParser = ArchiveParser(articlesFile, 1000, addCitations = true, addToDatabase = false)
        citationsParser.parse()
        val parsedArticleCitations = citationsParser.currentArticles
        checkArticles(parsedArticleCitations, correctArticles, citations = true)

    }

    @Test
    fun testParseArticles() {
        val listArticles = listOf(SemanticScholarArticles.article1,
                SemanticScholarArticles.article2, SemanticScholarArticles.article3,
                SemanticScholarArticles.article4, SemanticScholarArticles.article5)

        testArticlesForFile("articlesSemanticScholar.json", listArticles)
    }
}