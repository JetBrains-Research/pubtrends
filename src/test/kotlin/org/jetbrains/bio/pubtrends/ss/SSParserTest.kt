package org.jetbrains.bio.pubtrends.ss

import org.junit.Test
import kotlin.test.assertEquals

class SSParserTest {
    companion object {
        private const val testJSONFileName = "articlesSemanticScholar.json.gz"
        private lateinit var parser: ArchiveParser

        init {
            this::class.java.classLoader.getResourceAsStream(testJSONFileName).use {
                val file = createTempFile()
                file.outputStream().use { out ->
                    it.copyTo(out)
                }
                parser = ArchiveParser(file, 1000, addToDatabase = false, curFile = 1, filesAmount = 1)
                parser.parse()

            }
        }

        val parsedArticles = parser.currentArticles
    }

    @Test
    fun testParseArticlesCount() {
        assertEquals(SemanticScholarArticles.articles.size, parsedArticles.size)
    }

    @Test
    fun testParseArticleSSIDs() {
        assertEquals(SemanticScholarArticles.articles.map { it.ssid }, parsedArticles.map { it.ssid })
    }

    @Test
    fun testParseYear() {
        listOf(0, 1, 2).forEach {
            assertEquals(SemanticScholarArticles.articles[it].year, parsedArticles[it].year)
        }
    }

    @Test
    fun testParseNullYear() {
        listOf(3, 4).forEach {
            assertEquals(null, parsedArticles[it].year)
        }
    }

    @Test
    fun testParsePMID() {
        listOf(0, 1, 2, 3).forEach {
            assertEquals(SemanticScholarArticles.articles[it].pmid, parsedArticles[it].pmid)
        }
    }

    @Test
    fun testParseNullPMID() {
        assertEquals(null, parsedArticles[4].pmid)
    }

    @Test
    fun testParseTitle() {
        (0..4).forEach {
            assertEquals(SemanticScholarArticles.articles[it].title, parsedArticles[it].title)
        }
    }

    @Test
    fun testParseCitations() {
        listOf(0, 1, 2).forEach {
            assertEquals(SemanticScholarArticles.articles[it].citationList, parsedArticles[it].citationList)
        }
    }

    @Test
    fun testParseEmptyCitations() {
        listOf(3, 4).forEach {
            assertEquals(SemanticScholarArticles.articles[it].citationList, listOf())
        }
    }

    @Test
    fun testParseAbstract() {
        listOf(0, 1, 2).forEach {
            assertEquals(SemanticScholarArticles.articles[it].abstract, parsedArticles[it].abstract)
        }
    }

    @Test
    fun testParseNullAbstract() {
        listOf(3, 4).forEach {
            assertEquals(null, parsedArticles[it].abstract)
        }
    }

    @Test
    fun testParseDoi() {
        assertEquals(SemanticScholarArticles.articles[0].doi, parsedArticles[0].doi)
    }

    @Test
    fun testParseNullDoi() {
        listOf(1, 2, 3, 4).forEach {
            assertEquals(null, parsedArticles[it].doi)
        }
    }

    @Test
    fun testParseArxivSource() {
        assertEquals(SemanticScholarArticles.articles[4].source, PublicationSource.Arxiv)
    }

    @Test
    fun testParseNullSource() {
        listOf(0, 1, 2, 3).forEach {
            assertEquals(null, parsedArticles[it].source)
        }
    }

    @Test
    fun testParseKeywords() {
        listOf(1, 2).forEach {
            assertEquals(SemanticScholarArticles.articles[it].keywords, parsedArticles[it].keywords)
        }
    }

    @Test
    fun testParseEmptyKeywords() {
        listOf(0, 3, 4).forEach {
            assertEquals(null, parsedArticles[it].keywords)
        }
    }

    @Test
    fun testParseAuthorName() {
        (0..4).forEach() {
            assertEquals(SemanticScholarArticles.articles[it].aux.authors.map { author -> author.name },
                    parsedArticles[it].aux.authors.map { author -> author.name })
        }
    }

    @Test
    fun testParseJournalName() {
        (0..4).forEach() {
            assertEquals(SemanticScholarArticles.articles[it].aux.journal.name,
                    parsedArticles[it].aux.journal.name)
        }
    }

    @Test
    fun testParseJournalVolume() {
        (0..4).forEach() {
            assertEquals(SemanticScholarArticles.articles[it].aux.journal.volume,
                    parsedArticles[it].aux.journal.volume)
        }
    }

    @Test
    fun testParseJournalPages() {
        (0..4).forEach() {
            assertEquals(SemanticScholarArticles.articles[it].aux.journal.pages,
                    parsedArticles[it].aux.journal.pages)
        }
    }

    @Test
    fun testParseVenue() {
        (0..4).forEach() {
            assertEquals(SemanticScholarArticles.articles[it].aux.venue,
                    parsedArticles[it].aux.venue)
        }
    }

    @Test
    fun testParseLinks() {
        (0..4).forEach() {
            assertEquals(SemanticScholarArticles.articles[it].aux.links,
                    parsedArticles[it].aux.links)
        }
    }
}