package org.jetbrains.bio.pubtrends.pm

import org.junit.AfterClass
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ParserTest {
    companion object {
        private val dbHandler = MockDBHandler()
        private val parser = PubmedXMLParser(dbHandler, 0, 1000)
        private val testXMLFileName = "articlesForParserTest.xml"

        init {
            this::class.java.classLoader.getResourceAsStream(testXMLFileName)?.let {
                val file = createTempFile()
                file.outputStream().use { out ->
                    it.copyTo(out)
                }
                parser.parse(file.absolutePath)
            }
        }

        private val articleMap = parser.articleList.associate { Pair(it.pmid, it) }

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
        assertEquals(Articles.articles.size, parser.articleList.size)
    }

    @Test
    fun testParseArticlePMIDs() {
        assertEquals(Articles.articles.keys, parser.articleList.map { it.pmid }.toSet())
    }

    @Test
    fun testParseYear() {
        assertEquals(Articles.articles[420880]?.year, articleMap[420880]?.year)
    }

    @Test
    fun testParseYearFromMedlineDateField() {
        listOf(10188493, 14316043, 18122624).forEach {
            assertEquals(Articles.articles[it]?.year, articleMap[it]?.year)
        }
    }

    @Test
    fun testParsePublicationType() {
        listOf(420880, 11540070, 11243089).forEach {
            assertEquals(Articles.articles[it]?.type, articleMap[it]?.type)
        }
    }

    @Test
    fun testParseArticleTitleWithSpecialSymbols() {
        assertEquals(Articles.articles[420880]?.title, articleMap[420880]?.title)
    }

    @Test
    fun testParseArticleTitleWithInnerXMLTags() {
        assertEquals(Articles.articles[29391692]?.title, articleMap[29391692]?.title)
    }

    @Test
    fun testParseNoAbstract() {
        assertEquals("", articleMap[420880]?.abstractText)
    }

    @Test
    fun testParseAbstractWithInnerXMLTags() {
        listOf(29736257, 29456534).forEach {
            assertEquals(Articles.articles[it]?.abstractText, articleMap[it]?.abstractText)
        }
    }

    @Test
    fun testParseStructuredAbstract() {
        listOf(20453483, 27654823, 24884411).forEach {
            assertEquals(Articles.articles[it]?.abstractText, articleMap[it]?.abstractText)
        }
    }

    @Test
    fun testParseOtherAbstractField() {
        listOf(11243089, 11540070).forEach {
            assertEquals(Articles.articles[it]?.abstractText, articleMap[it]?.abstractText)
        }
    }

    @Test
    fun testParseNoKeywords() {
        assertTrue(articleMap[420880]?.keywordList?.isEmpty() ?: false)
    }

    @Test
    fun testParseKeywordCount() {
        assertEquals(Articles.articles[29456534]?.keywordList?.size, articleMap[29456534]?.keywordList?.size)
    }

    @Test
    fun testParseKeywords() {
        listOf(29456534, 14316043, 18122624).forEach {
            assertEquals(Articles.articles[it]?.keywordList, articleMap[it]?.keywordList)
        }
    }

    @Test
    fun testParseNoMeSH() {
        assertTrue(articleMap[29391692]?.meshHeadingList?.isEmpty() ?: false)
    }

    @Test
    fun testParseMeSHCount() {
        assertEquals(Articles.articles[420880]?.meshHeadingList?.size, articleMap[420880]?.meshHeadingList?.size)
    }

    @Test
    fun testParseMeSH() {
        assertEquals(Articles.articles[420880]?.meshHeadingList, articleMap[420880]?.meshHeadingList)
    }

    @Test
    fun testParseNoCitations() {
        assertTrue(articleMap[420880]?.citationList?.isEmpty() ?: false)
    }

    @Test
    fun testParseCitationCount() {
        assertEquals(Articles.articles[24884411]?.citationList?.size, articleMap[24884411]?.citationList?.size)
    }

    @Test
    fun testParseCitationPMIDs() {
        assertEquals(Articles.articles[24884411]?.citationList, articleMap[24884411]?.citationList)
    }

    @Test
    fun testParseNoDatabanks() {
        assertTrue(Articles.articles[24884411]?.auxInfo?.databanks?.isEmpty() ?: false)
    }

    @Test
    fun testParseDatabankCount() {
        assertEquals(Articles.articles[420880]?.auxInfo?.databanks?.size, articleMap[420880]?.auxInfo?.databanks?.size)
    }

    @Test
    fun testParseDatabankNames() {
        assertEquals(Articles.articles[420880]?.auxInfo?.databanks?.map { it.name },
            articleMap[420880]?.auxInfo?.databanks?.map { it.name })
    }

    @Test
    fun testParseAccessionNumberCount() {
        assertEquals(Articles.articles[420880]?.auxInfo?.databanks?.map { it.accessionNumber.size },
            articleMap[420880]?.auxInfo?.databanks?.map { it.accessionNumber.size })
    }

    @Test
    fun testParseAccessionNumbers() {
        assertEquals(Articles.articles[420880]?.auxInfo?.databanks?.map { it.accessionNumber },
            articleMap[420880]?.auxInfo?.databanks?.map { it.accessionNumber })
    }

    @Test
    fun testParseNoAuthors() {
        assertTrue(Articles.articles[420880]?.auxInfo?.authors?.isEmpty() ?: false)
    }

    @Test
    fun testParseAuthorCount() {
        assertEquals(
            Articles.articles[29456534]?.auxInfo?.authors?.size,
            articleMap[29456534]?.auxInfo?.authors?.size
        )
    }

    @Test
    fun testParseAuthorNames() {
        assertEquals(Articles.articles[29456534]?.auxInfo?.authors?.map { it.name },
            articleMap[29456534]?.auxInfo?.authors?.map { it.name })
    }

    @Test
    fun testParseAuthorAffiliationCount() {
        assertEquals(Articles.articles[29456534]?.auxInfo?.authors?.map { it.affiliation.size },
            articleMap[29456534]?.auxInfo?.authors?.map { it.affiliation.size })
    }

    @Test
    fun testParseAuthorAffiliations() {
        assertEquals(Articles.articles[29456534]?.auxInfo?.authors?.map { it.affiliation },
            articleMap[29456534]?.auxInfo?.authors?.map { it.affiliation })
    }

    @Test
    fun testParseJournal() {
        assertEquals(Articles.articles[420880]?.auxInfo?.journal?.name, articleMap[420880]?.auxInfo?.journal?.name)
    }

    @Test
    fun testParseLanguage() {
        assertEquals(Articles.articles[420880]?.auxInfo?.language, articleMap[420880]?.auxInfo?.language)
    }

    @Test
    fun testParseNoDOI() {
        assertEquals("", articleMap[420880]?.doi)
    }

    @Test
    fun testParseDOI() {
        assertEquals(Articles.articles[29391692]?.doi, articleMap[29391692]?.doi)
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