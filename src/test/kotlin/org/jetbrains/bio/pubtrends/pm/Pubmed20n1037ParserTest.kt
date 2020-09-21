package org.jetbrains.bio.pubtrends.pm

import org.jetbrains.bio.pubtrends.db.MockDBWriter
import org.junit.Test
import kotlin.test.assertEquals

/**
 * Test case for incorrect Pubmed XML description
 * See https://github.com/JetBrains-Research/pubtrends/issues/203
 */
class Pubmed20n1037ParserTest {
    private val dbHandler = MockDBWriter<PubmedArticle>()
    private val parser = PubmedXMLParser(dbHandler, 1000)

    @Test
    fun testParse() {
        parser.parse(this::class.java.classLoader.getResource("updatePubmed20n1037.xml").toURI().path)
        assertEquals(1, dbHandler.articlesStored)
    }
}

