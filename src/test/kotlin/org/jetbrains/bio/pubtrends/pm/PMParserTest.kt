package org.jetbrains.bio.pubtrends.pm

import org.jetbrains.bio.pubtrends.MockDBHandler
import org.junit.AfterClass
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PMParserTest {
    companion object {
        private val dbHandler = MockDBHandler<PubmedArticle>()
        private val parser = PubmedXMLParser(dbHandler, 0, 1000)
        private const val testXMLFileName = "articlesForParserTest.xml"

        init {
            this::class.java.classLoader.getResourceAsStream(testXMLFileName)?.let {
                val file = createTempFile()
                file.outputStream().use { out ->
                    it.copyTo(out)
                }
                parser.parse(file.absolutePath)
            }
        }

        private val articleMap = parser.articleList.associateBy { it.pmid }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            val testFile = File(testXMLFileName)
            testFile.delete()
            assertFalse(testFile.exists())
        }
    }

    @Test
    fun testParseArticlesCount() {
        assertEquals(PubmedArticles.articles.size, parser.articleList.size)
    }

    @Test
    fun testParseArticlePMIDs() {
        assertEquals(PubmedArticles.articles.keys, parser.articleList.map { it.pmid }.toSet())
    }

    @Test
    fun testParseYear() {
        assertEquals(PubmedArticles.articles[420880]?.date, articleMap[420880]?.date)
    }

    @Test
    fun testParseYearFromMedlineDateField() {
        listOf(10188493, 14316043, 18122624).forEach {
            assertEquals(PubmedArticles.articles[it]?.date, articleMap[it]?.date)
        }
    }

    @Test
    fun testParsePublicationType() {
        listOf(420880, 11540070, 11243089).forEach {
            assertEquals(PubmedArticles.articles[it]?.type, articleMap[it]?.type)
        }
    }

    @Test
    fun testParseArticleTitleWithSpecialSymbols() {
        assertEquals(PubmedArticles.articles[420880]?.title, articleMap[420880]?.title)
    }

    @Test
    fun testParseArticleTitleWithInnerXMLTags() {
        assertEquals(PubmedArticles.articles[29391692]?.title, articleMap[29391692]?.title)
    }

    @Test
    fun testParseNoAbstract() {
        assertEquals("", articleMap[420880]?.abstractText)
    }

    @Test
    fun testParseAbstractWithInnerXMLTags() {
        listOf(29736257, 29456534).forEach {
            assertEquals(PubmedArticles.articles[it]?.abstractText, articleMap[it]?.abstractText)
        }
    }

    @Test
    fun testParseStructuredAbstract() {
        listOf(20453483, 27654823, 24884411).forEach {
            assertEquals(PubmedArticles.articles[it]?.abstractText, articleMap[it]?.abstractText)
        }
    }

    @Test
    fun testParseOtherAbstractField() {
        listOf(11243089, 11540070).forEach {
            assertEquals(PubmedArticles.articles[it]?.abstractText, articleMap[it]?.abstractText)
        }
    }

    @Test
    fun testParseNoKeywords() {
        assertTrue(articleMap[420880]?.keywordList?.isEmpty() ?: false)
    }

    @Test
    fun testParseKeywordCount() {
        assertEquals(PubmedArticles.articles[29456534]?.keywordList?.size, articleMap[29456534]?.keywordList?.size)
    }

    @Test
    fun testParseKeywords() {
        listOf(29456534, 14316043, 18122624).forEach {
            assertEquals(PubmedArticles.articles[it]?.keywordList, articleMap[it]?.keywordList)
        }
    }

    @Test
    fun testParseNoMeSH() {
        assertTrue(articleMap[29391692]?.meshHeadingList?.isEmpty() ?: false)
    }

    @Test
    fun testParseMeSHCount() {
        assertEquals(PubmedArticles.articles[420880]?.meshHeadingList?.size, articleMap[420880]?.meshHeadingList?.size)
    }

    @Test
    fun testParseMeSH() {
        assertEquals(PubmedArticles.articles[420880]?.meshHeadingList, articleMap[420880]?.meshHeadingList)
    }

    @Test
    fun testParseNoCitations() {
        assertTrue(articleMap[420880]?.citationList?.isEmpty() ?: false)
    }

    @Test
    fun testParseCitationCount() {
        assertEquals(PubmedArticles.articles[24884411]?.citationList?.size, articleMap[24884411]?.citationList?.size)
    }

    @Test
    fun testParseCitationPMIDs() {
        assertEquals(PubmedArticles.articles[24884411]?.citationList, articleMap[24884411]?.citationList)
    }

    @Test
    fun testParseNoDatabanks() {
        assertTrue(PubmedArticles.articles[24884411]?.auxInfo?.databanks?.isEmpty() ?: false)
    }

    @Test
    fun testParseDatabankCount() {
        assertEquals(PubmedArticles.articles[420880]?.auxInfo?.databanks?.size, articleMap[420880]?.auxInfo?.databanks?.size)
    }

    @Test
    fun testParseDatabanks() {
        assertEquals(PubmedArticles.articles[420880]?.auxInfo?.databanks?.map { it.name },
                articleMap[420880]?.auxInfo?.databanks?.map { it.name })
        assertEquals(PubmedArticles.articles[420880]?.auxInfo?.databanks?.map { it.accessionNumber },
                articleMap[420880]?.auxInfo?.databanks?.map { it.accessionNumber })
    }

    @Test
    fun testParseAccessionNumbers() {
        assertEquals(PubmedArticles.articles[420880]?.auxInfo?.databanks?.map { it.accessionNumber },
                articleMap[420880]?.auxInfo?.databanks?.map { it.accessionNumber })
    }

    @Test
    fun testParseNoAuthors() {
        assertTrue(PubmedArticles.articles[420880]?.auxInfo?.authors?.isEmpty() ?: false)
    }

    @Test
    fun testParseAuthorCount() {
        assertEquals(
                PubmedArticles.articles[29456534]?.auxInfo?.authors?.size,
                articleMap[29456534]?.auxInfo?.authors?.size
        )
    }

    @Test
    fun testParseAuthorNames() {
        assertEquals(PubmedArticles.articles[29456534]?.auxInfo?.authors?.map { it.name },
                articleMap[29456534]?.auxInfo?.authors?.map { it.name })
    }

    @Test
    fun testParseAuthorCollectiveName() {
        assertEquals(PubmedArticles.articles[621131]?.auxInfo?.authors?.map { it.name },
                articleMap[621131]?.auxInfo?.authors?.map { it.name })
    }

    @Test
    fun testParseAuthorAffiliationCount() {
        assertEquals(PubmedArticles.articles[29456534]?.auxInfo?.authors?.map { it.affiliation.size },
                articleMap[29456534]?.auxInfo?.authors?.map { it.affiliation.size })
    }

    @Test
    fun testParseAuthorAffiliations() {
        assertEquals(PubmedArticles.articles[29456534]?.auxInfo?.authors?.map { it.affiliation },
                articleMap[29456534]?.auxInfo?.authors?.map { it.affiliation })
    }
// TODO revert once switching to full Pubmed Neo4j model
//    @Test
//    fun testParseAuthorAffiliationsIgnoreEmpty() {
//        assertTrue(Articles.articles[24884411]?.auxInfo?.authors?.first()?.affiliation?.isEmpty()!!)
//    }

    @Test
    fun testParseAuthorAffiliationsSplitMultiple() {
        assertEquals(PubmedArticles.articles[27560010]?.auxInfo?.authors?.map { it.affiliation },
                articleMap[27560010]?.auxInfo?.authors?.map { it.affiliation })
    }

// TODO revert once switching to full Pubmed Neo4j model
//    @Test
//    fun testParseAuthorAffiliationsWithInnerXMLTags() {
//        assertEquals(Articles.articles[24884411]?.auxInfo?.authors?.map { it.affiliation },
//                articleMap[24884411]?.auxInfo?.authors?.map { it.affiliation })
//    }

    @Test
    fun testParseJournal() {
        assertEquals(PubmedArticles.articles[420880]?.auxInfo?.journal?.name, articleMap[420880]?.auxInfo?.journal?.name)
    }

    @Test
    fun testParseLanguage() {
        assertEquals(PubmedArticles.articles[420880]?.auxInfo?.language, articleMap[420880]?.auxInfo?.language)
    }

    @Test
    fun testParseNoDOI() {
        assertEquals("", articleMap[420880]?.doi)
    }

    @Test
    fun testParseDOI() {
        assertEquals(PubmedArticles.articles[29391692]?.doi, articleMap[29391692]?.doi)
    }

    @Test
    fun testParseDeleteArticlesCount() {
        assertEquals(2, parser.deletedArticlePMIDList.size)
    }

    @Test
    fun testParseDeleteArticlePMIDs() {
        assertEquals(listOf(12345, 23456), parser.deletedArticlePMIDList)
    }
}