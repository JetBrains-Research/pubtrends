package org.jetbrains.bio.pubtrends.pm

import org.junit.AfterClass
import org.junit.BeforeClass
import org.junit.Test
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DBHandlerTest {
    companion object {
        private val settingsRoot: Path = Paths.get(System.getProperty("user.home", ""), ".pubtrends")

        init {
            check(Files.exists(settingsRoot)) {
                "$settingsRoot should have been created by log4j"
            }
        }

        private val configPath: Path = settingsRoot.resolve("config.properties")

        init {
            check(Files.exists(configPath)) {
                "Config file not found, please modify and copy config.properties to $configPath"
            }
        }

        private val config = Properties().apply {
            load(BufferedReader(FileReader(configPath.toFile())))
        }

        private val dbHandler = TestDBHandler(
            config["test_url"].toString(),
            config["test_port"].toString().toInt(),
            config["test_database"].toString(),
            config["test_username"].toString(),
            config["test_password"].toString(),
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

        @BeforeClass
        @JvmStatic
        fun setUp() {
            val path = parserFileSetup(path)
            check(path != "") {
                "Failed to load test file: $path"
            }
            parser.parse(path)
        }

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
        assertEquals(parser.articleList.size, dbHandler.articlesCount)
    }

    @Test
    fun testStoreArticlePMIDs() {
        assertEquals(parser.articleList.map { it.pmid }, dbHandler.articlesPMIDList)
    }
}