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

    @Test
    fun testParseForArticleWithPlainAbstract() {
        val path = parserFileSetup("article29736257withPlainAbstract.xml")
        crawler.parse(path)

        val articles = crawler.pubmedXMLHandler.articles
        val expectedTitle = "Evidence that S6K1, but not 4E-BP1, mediates skeletal muscle " +
                "pathology associated with loss of A-type lamins."
        val expectedAbstract = "The mechanistic target of rapamycin (mTOR) signaling pathway plays " +
                "a central role in aging and a number of different disease states. Rapamycin, which " +
                "suppresses activity of the mTOR complex 1 (mTORC1), shows preclinical (and sometimes " +
                "clinical) efficacy in a number of disease models."

        assertEquals(1, articles.size, "Wrong number of articles")
        assertEquals(29736257, articles[0].pmid,  "Wrong PMID")
        assertEquals(2017, articles[0].year, "Wrong publication year")
        assertEquals(expectedTitle, articles[0].title, "Wrong title")
        assertEquals(expectedAbstract, articles[0].abstractText, "Wrong abstract")

        val keywords = articles[0].keywordList
        val expectedKeywordList = listOf("4E-BP1", "Lmna−/− mice", "S6K1", "lifespan", "mTORC1", "muscle", "rapamycin")
        assertEquals(expectedKeywordList.size, keywords.size,"Wrong number of keywords")
        assertEquals(expectedKeywordList, keywords,"Wrong keywords in the list")

        val citations = articles[0].citationList
        val expectedCitationList = listOf(10587585, 11855819, 26614871, 12702809)
        assertEquals(expectedCitationList.size, citations.size,"Wrong number of citations")
        assertEquals(expectedCitationList, citations, "Wrong citations in the list")
    }

    /*@Test
    fun testParseForArticleWithFormattedAbstract() {
        val path = parserFileSetup("article29736257withFormattedAbstract.xml")
        crawler.parse(path)

        val articles = crawler.pubmedXMLHandler.articles
        val expectedTitle = "Evidence that S6K1, but not 4E-BP1, mediates skeletal muscle " +
                "pathology associated with loss of A-type lamins."
        val expectedAbstract = "The mechanistic target of rapamycin (mTOR) signaling pathway plays a central " +
                "role in aging and a number of different disease states. Rapamycin, which suppresses activity" +
                " of the mTOR complex 1 (mTORC1), shows preclinical (and sometimes clinical) efficacy in a" +
                " number of disease models. Among these are Lmna-/-mice, which serve as a mouse model " +
                "for dystrophy-associated laminopathies."

        assertEquals(1, articles.size, "Wrong number of articles")
        assertEquals(29736257, articles[0].pmid,  "Wrong PMID")
        assertEquals(2017, articles[0].year, "Wrong publication year")
        assertEquals(expectedTitle, articles[0].title, "Wrong title")
        assertEquals(expectedAbstract, articles[0].abstractText, "Wrong abstract")

        val keywords = articles[0].keywordList
        val expectedKeywordList = listOf("4E-BP1", "Lmna−/− mice", "S6K1", "lifespan", "mTORC1", "muscle", "rapamycin")
        assertEquals(expectedKeywordList.size, keywords.size,"Wrong number of keywords")
        assertEquals(expectedKeywordList, keywords,"Wrong keywords in the list")

        val citations = articles[0].citationList
        val expectedCitationList = listOf(10587585, 11855819, 26614871, 12702809)
        assertEquals(expectedCitationList.size, citations.size,"Wrong number of citations")
        assertEquals(expectedCitationList, citations, "Wrong citations in the list")
    }*/
}