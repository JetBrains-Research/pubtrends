package org.jetbrains.bio.pubtrends.arxiv

import com.github.kittinunf.fuel.*
import com.github.kittinunf.fuel.core.FuelError
import com.github.kittinunf.fuel.core.Request
import com.github.kittinunf.fuel.core.Response
import com.github.kittinunf.result.Result
import io.mockk.*
import org.apache.logging.log4j.kotlin.KotlinLogger
import org.junit.jupiter.api.*

import org.junit.jupiter.api.Assertions.*

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ArxivAPITests {

    @BeforeAll
    fun initTest() {
        mockkConstructor(KotlinLogger::class)
        every {anyConstructed<KotlinLogger>().info(any<String>())} just Runs
    }

    @AfterAll
    fun clear() {
        unmockkAll()
    }

    @Test
    fun testGetBulkApiRecordsWithResumptionTokenSuccess() {
        val spyArxiApi = spyk(ArxivAPI)
        val requestMock = mockk<Request>()
        val resultMock = mockk<Result.Success<String>>()
        val responseMock = mockk<Response>()

        val url = ArxivAPI.requestBulkUrlPrefix + "verb=ListRecords&resumptionToken=" + "token1111"
        mockkStatic("com.github.kittinunf.fuel.FuelKt")
        every { url.httpGet().timeoutRead(any()).responseString() }returns
                Triple(requestMock, responseMock, resultMock)

        every { resultMock.get()} returns "xml text"

        mockkObject(ArxivXMLDomParser)
        every { ArxivXMLDomParser.parseArxivRecords("xml text")} returns
                Triple(listOf(ArxivData("ident", id = "id1")), "new token", 1000)
        val slot = slot<List<String>>()
        every { spyArxiApi.getRecordsUrl(capture(slot))} answers {listOf("pdf url ${slot.captured[0]}")}


        val (arxivRecords, newResToken, recTotal) =
            spyArxiApi.getBulkArxivRecords("", "token1111", 100)

        assertTrue(arxivRecords.size == 1)
        assertEquals("id1", arxivRecords.get(0).id)
        assertEquals("pdf url id1", arxivRecords.get(0).pdfUrl)
        assertEquals("new token", newResToken)
        assertEquals(1000, recTotal)
    }

    @Test
    fun testGetBulkApiRecordsWithDateSuccess() {
        val spyArxiApi = spyk(ArxivAPI)
        val requestMock = mockk<Request>()
        val resultMock = mockk<Result.Success<String>>()
        val responseMock = mockk<Response>()

        val url = ArxivAPI.requestBulkUrlPrefix + "verb=ListRecords&from=2020-01-10&metadataPrefix=arXiv"
        mockkStatic("com.github.kittinunf.fuel.FuelKt")
        every { url.httpGet().timeoutRead(any()).responseString() }returns
                Triple(requestMock, responseMock, resultMock)

        every { resultMock.get()} returns "xml text"

        mockkObject(ArxivXMLDomParser)
        every { ArxivXMLDomParser.parseArxivRecords("xml text")} returns
                Triple(listOf(ArxivData("ident", id = "id1")), "new token", 1000)
        val slot = slot<List<String>>()
        every { spyArxiApi.getRecordsUrl(capture(slot))} answers {listOf("pdf url ${slot.captured[0]}")}


        val (arxivRecords, newResToken, recTotal) =
            spyArxiApi.getBulkArxivRecords("2020-01-10", "", 100)

        assertTrue(arxivRecords.size == 1)
        assertEquals("id1", arxivRecords.get(0).id)
        assertEquals("pdf url id1", arxivRecords.get(0).pdfUrl)
        assertEquals("new token", newResToken)
        assertEquals(1000, recTotal)
    }

    @Test
    fun testGetBulkApiRecordsFailure() {
        val spyArxiApi = spyk(ArxivAPI, recordPrivateCalls = true)
        val requestMock = mockk<Request>()
        val resultFailMock = mockk<Result.Failure<FuelError>>()
        val resultSuccessMock = mockk<Result.Success<String>>()
        val responseMock = mockk<Response>()

        val url = ArxivAPI.requestBulkUrlPrefix + "verb=ListRecords&from=2020-01-10&metadataPrefix=arXiv"
        mockkStatic("com.github.kittinunf.fuel.FuelKt")
        every { url.httpGet().timeoutRead(any()).responseString() } returnsMany
                listOf(Triple(requestMock, responseMock, resultFailMock),
                       Triple(requestMock, responseMock, resultSuccessMock)
                )
        spyArxiApi.sleepTime = 20
        every { resultFailMock.getException()} returns mockk()
        every { responseMock.statusCode } returns 503

        every { resultSuccessMock.get()} returns "xml text"

        mockkObject(ArxivXMLDomParser)
        every { ArxivXMLDomParser.parseArxivRecords("xml text")} returns
                Triple(listOf(ArxivData("ident", id = "id1")), "new token", 1000)
        val slot = slot<List<String>>()
        every { spyArxiApi.getRecordsUrl(capture(slot))} answers {listOf("pdf url ${slot.captured[0]}")}


        val (arxivRecords, newResToken, recTotal) =
            spyArxiApi.getBulkArxivRecords("2020-01-10", "", 100)

        assertTrue(arxivRecords.size == 1)
        assertEquals("id1", arxivRecords.get(0).id)
        assertEquals("pdf url id1", arxivRecords.get(0).pdfUrl)
        assertEquals("new token", newResToken)
        assertEquals(1000, recTotal)
    }

    @Test
    fun testGetBulkApiRecordsException() {
        val spyArxiApi = spyk(ArxivAPI, recordPrivateCalls = true)
        val requestMock = mockk<Request>()
        val resultFailMock = mockk<Result.Failure<FuelError>>()
        val responseMock = mockk<Response>()

        val url = ArxivAPI.requestBulkUrlPrefix + "verb=ListRecords&from=2020-01-10&metadataPrefix=arXiv"
        mockkStatic("com.github.kittinunf.fuel.FuelKt")
        every { url.httpGet().timeoutRead(any()).responseString() } returns
                Triple(requestMock, responseMock, resultFailMock)
        val fmock = mockk<FuelError>()
        every { fmock.message } returns "error"
        every { resultFailMock.getException()} returns fmock
        every { responseMock.statusCode } returns 404

        assertThrows(
            ArxivAPI.ApiRequestFailedException::class.java
        ) {spyArxiApi.getBulkArxivRecords("2020-01-10", "", 100)}
    }

}