package org.jetbrains.bio.pubtrends.pm

import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.assertEquals

@RunWith(Parameterized::class)
class BatchProcessingTest(private val batchSize : Int) {

    @Test
    fun testBatchSize() {
        val parser = PubmedXMLParser(dbHandler, 0, batchSize)

        val testXMLFileName = "articlesForParserTest.xml"
        this::class.java.classLoader.getResourceAsStream(testXMLFileName)?.let {
            val file = createTempFile()
            file.outputStream().use {out ->
                it.copyTo(out)
            }
            parser.parse(file.absolutePath)
        }

        assertEquals(12, dbHandler.articlesStored)
        dbHandler.articlesStored = 0
    }

    companion object {
        private val dbHandler = MockDBHandler(batch = true)

        @Parameterized.Parameters(name = "batchSize={0}")
        @JvmStatic fun data() = listOf(
                0,  // default value - save all at once
                3,  // 12 % 3 = 0
                5,  // 12 % 5 != 0
                12, // equal to the amount of articles
                20  // more than amount of articles
        )
    }
}