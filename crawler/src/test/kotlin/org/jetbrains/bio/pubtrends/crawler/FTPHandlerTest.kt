package org.jetbrains.bio.pubtrends.crawler

import org.junit.Test
import kotlin.test.assertEquals

class FTPHandlerTest {
    private val handler = MockFTPHandler()

    /*@Test
    fun testFetchDefaultArgs() {
        val files = handler.fetch()
        val baselineFiles = files.first
        val updateFiles = files.second

        assertEquals(listOf("article29736257withPlainAbstract.xml.gz"), baselineFiles,
                "Wrong baseline files in the list")
        assertEquals(listOf("articlesWithFormattedAbstract.xml.gz"), updateFiles,
                "Wrong update files in the list")
    }*/

    /*@Test
    fun testFetchWithTimestamp() {
        // Timestamp to get only update file
        val lastCheck: Long = 1541203200000

        val files = handler.fetch(lastCheck)
        val baselineFiles = files.first
        val updateFiles = files.second

        assertEquals(0, baselineFiles.size, "Baseline file was not supposed to be found")
        assertEquals(listOf("articlesWithFormattedAbstract.xml.gz"), updateFiles,
                "Wrong update files in the list")
    }*/
}