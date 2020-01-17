package org.jetbrains.bio.pubtrends.biorxiv

import org.junit.Test
import kotlin.test.assertEquals

class BiorxivScraperTest {

    private val scraper = BiorxivScraper()

    @Test
    fun testGetIdOldType() {
        assertEquals(123456, scraper.getId("123456"))
    }

    @Test
    fun testGetIdNewType() {
        assertEquals(123456, scraper.getId("2020.01.18.123456"))
    }
}