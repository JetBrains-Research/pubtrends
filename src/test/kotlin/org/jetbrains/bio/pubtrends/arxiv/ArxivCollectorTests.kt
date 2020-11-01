package org.jetbrains.bio.pubtrends.arxiv

import org.jetbrains.bio.pubtrends.neo4j.DatabaseHandler
import org.jetbrains.bio.pubtrends.ref.GrobidEngine
import io.mockk.*
import org.apache.logging.log4j.kotlin.KotlinLogger
import org.junit.Assert.assertEquals
import org.junit.jupiter.api.*

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ArxivCollectorTests {

    @BeforeAll
    fun initTest() {
        mockkConstructor(KotlinLogger::class)
        every {anyConstructed<KotlinLogger>().info(any<String>())} just Runs
        mockkObject(ArxivPDFHandler)
        every { ArxivPDFHandler.getFullInfo(any(), any(), any(), any(), any(), any()) } just Runs
        mockkObject(GrobidEngine)
        mockkObject(ArxivAPI)
    }

    @AfterAll
    fun clear() {
        unmockkAll()
    }

    @Test
    fun emptyResumptionTokenTest() {
        val records = mutableListOf<ArxivData>()
        val dbMock = mockk<DatabaseHandler>()
        val slot = slot<List<ArxivData>>()
        every { dbMock.storeArxivData(capture(slot)) } answers {records += slot.captured}
        val arxivList1 = listOf(
            ArxivData(identifier = "id1", id = "1", title = "title1"),
            ArxivData(identifier = "id2", id = "2", title = "title2")
        )
        val arxivList2 = listOf(
            ArxivData(identifier = "id3", id = "3", title = "title3")
        )
        val arxivList3 = listOf(
            ArxivData(identifier = "id4", id = "4", title = "title4")
        )
        every { ArxivAPI.getBulkArxivRecords(any(), "", any())} returns
                Triple(arxivList1, "restoken1", 1000)
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken1", any())} returns
                Triple(arxivList2, "restoken2", 1000)
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken2", any())} returns
                Triple(arxivList3, "", 1000)
        val expectedList = arxivList1 + arxivList2 + arxivList3

        ArxivCollector.collect("2018-04-10",
            dbMock,
            listOf(),
            ""
        )

        assertEquals(expectedList, records)
    }

    @Test
    fun nonEmptyResumptionTokenTest() {
        val records = mutableListOf<ArxivData>()
        val dbMock = mockk<DatabaseHandler>()
        val slot = slot<List<ArxivData>>()
        every { dbMock.storeArxivData(capture(slot)) } answers {records += slot.captured}
        val arxivList1 = listOf(
            ArxivData(identifier = "id1", id = "1", title = "title1"),
            ArxivData(identifier = "id2", id = "2", title = "title2")
        )
        val arxivList2 = listOf(
            ArxivData(identifier = "id3", id = "3", title = "title3")
        )
        val arxivList3 = listOf(
            ArxivData(identifier = "id4", id = "4", title = "title4")
        )
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken0", any())} returns
                Triple(arxivList1, "restoken1", 1000)
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken1", any())} returns
                Triple(arxivList2, "restoken2", 1000)
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken2", any())} returns
                Triple(arxivList3, "", 1000)
        val expectedList = arxivList1 + arxivList2 + arxivList3

        ArxivCollector.collect("2018-04-10",
            dbMock,
            listOf(),
            "restoken0"
        )

        assertEquals(expectedList, records)
    }

    @Test
    fun exceptionTest() {
        val records = mutableListOf<ArxivData>()
        val dbMock = mockk<DatabaseHandler>()
        val slot = slot<List<ArxivData>>()
        every { dbMock.storeArxivData(capture(slot)) } answers {records += slot.captured}
        val arxivList1 = listOf(
            ArxivData(identifier = "id1", id = "1", title = "title1"),
            ArxivData(identifier = "id2", id = "2", title = "title2")
        )
        val arxivList2 = listOf(
            ArxivData(identifier = "id3", id = "3", title = "title3")
        )
        val arxivList3 = listOf(
            ArxivData(identifier = "id4", id = "4", title = "title4")
        )
        ArxivCollector.sleepTime = 1
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken0", any())} returns
                Triple(arxivList1, "restoken1", 1000)
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken1", any())} throws
                ArxivAPI.ApiRequestFailedException("") andThen Triple(arxivList2, "restoken2", 1000)
        every { ArxivAPI.getBulkArxivRecords(any(), "restoken2", any())} returns
                Triple(arxivList3, "", 1000)
        val expectedList = arxivList1 + arxivList2 + arxivList3

        ArxivCollector.collect("2018-04-10",
            dbMock,
            listOf(),
            "restoken0"
        )

        assertEquals(expectedList, records)
    }
}