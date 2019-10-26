package org.jetbrains.bio.pubtrends.pm

import org.jetbrains.bio.pubtrends.Config
import org.jetbrains.bio.pubtrends.EmbeddedNeo4jInstance
import org.junit.AfterClass
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Neo4jDBHandlerTest {
    companion object {
        init {
            EmbeddedNeo4jInstance.load()
        }

        // Load configuration file
        private val config = Config.config

        private val dbHandler = TestNeo4jDBHandler(
                config["test_neo4jurl"].toString(),
                config["test_neo4jport"].toString().toInt(),
                config["test_neo4jusername"].toString(),
                config["test_neo4jpassword"].toString(),
                resetDatabase = true
        )
        private val parser = PubmedXMLParser(dbHandler, 0, 1000)

        private const val path = "articlesForDBHandlerTest.xml"

        private fun parserFileSetup(name: String): String {
            this::class.java.classLoader.getResourceAsStream(name)?.use {
                val file = createTempFile()
                file.outputStream().use { out ->
                    it.copyTo(out)
                }
                return file.absolutePath
            }
            return ""
        }

        init {
            val path = parserFileSetup(path)
            check(path != "") {
                "Failed to load test file: $path"
            }
            parser.parse(path)
        }

        val articles = parser.articleList
        val articlePMIDs = articles.map { it.pmid }
        val articleReferencePMIDs = articles.flatMap { it.citationList }
        val totalArticles = (articlePMIDs + articleReferencePMIDs).sorted()

        @AfterClass
        @JvmStatic
        fun tearDown() {
            val testFile = File(path)
            testFile.delete()
            assertTrue { !testFile.exists() }
        }
    }

    @Test
    fun testStoreArticlesCount() {
        assertEquals(totalArticles.size, dbHandler.articlesCount)
    }

    @Test
    fun testStoreArticlePMIDs() {
        assertEquals(totalArticles, dbHandler.articlesPMIDList)
    }
}