package org.jetbrains.bio.pubtrends.arxiv

import org.jetbrains.bio.pubtrends.data.Author
import io.mockk.*
import org.apache.logging.log4j.kotlin.KotlinLogger
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ArxivXmlParserTests {
    @BeforeAll
    fun initTest() {
        mockkConstructor(KotlinLogger::class)
        every {anyConstructed<KotlinLogger>().info(any<String>())} just Runs
    }

    @AfterAll
    fun clear() {
        unmockkAll()
    }

    fun loadFromResourses(fileName : String) : String {
        return this.javaClass.getResource("/xmlParserTests/$fileName").readText()
    }

    @Test
    fun fullDataTest() {
        val xmlText = loadFromResourses("xmlArxivData.xml")
        val expectedResumptionToken = "4342418|1001"
        val expectedTotalRecords = 7002
        val expectedRecord1 = ArxivData(
            identifier = "oai:arXiv.org:0801.2407",
            datestamp = "2020-03-25",
            id = "0801.2407",
            creationDate = "2008-01-15",
            authors = mutableListOf(
                Author("Laetitia Abel-Tiberini II", "LAOG"),
                Author("Pierre Labeye", "CEA Leti")
            ),
            title = "ABCD ABCD DEFG",
            categories = mutableListOf("astro-ph"),
            comments = "This is comment",
            doi = "10.1364/OE.15.018005",
            abstract = "This is abstract",
            mscClass = "55p35"
        )

        val expectedRecord2 = ArxivData(
            identifier = "oai:arXiv.org:0909.2882",
            datestamp = "2020-03-24",
            id = "0909.2882",
            creationDate = "2009-09-15",
            lastUpdateDate = "2020-03-21",
            authors = mutableListOf(
                Author("Wm. G. Hoover"),
                Author("Carol G. Hoover")
            ),
            title = "Shockwaves and Local Hydrodynamics; Failure of the Navier-Stokes Equations",
            categories = mutableListOf("physics.flu-dyn", "nlin.CD"),
            comments = "many lines comment",
            doi = "10.1142/9789814307543_0002",
            license = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            abstract = "Big abstract abacaba DABA-DABA",
            acmClass = "56asdf",
            reportNo = "IGPG-07/03-2"
        )

        val (arxivRecords, resumptionToken, recordsTotal) = ArxivXMLDomParser.parseArxivRecords(xmlText)

        assertTrue(resumptionToken == expectedResumptionToken)
        assertTrue(recordsTotal == expectedTotalRecords)
        assertTrue(arxivRecords.size == 2)
        assertEquals(expectedRecord1, arxivRecords[0])
        assertEquals(expectedRecord2, arxivRecords[1])
    }

    @Test
    fun parseArxivRecordsWithMissingRequiredInfo() {
        val xmlText = loadFromResourses("misArxivData.xml")
        val expectedRecord = ArxivData(
            identifier = "oai:arXiv.org:0909.2882",
            datestamp = "2020-03-24",
            id = "0909.2882",
            creationDate = "2009-09-15",
            lastUpdateDate = "2020-03-21",
            authors = mutableListOf(
                Author("Wm. G. Hoover"),
                Author("Carol G. Hoover")
            ),
            title = "Shockwaves and Local Hydrodynamics; Failure of the Navier-Stokes Equations",
            categories = mutableListOf("physics.flu-dyn", "nlin.CD"),
            comments = "many lines comment",
            doi = "10.1142/9789814307543_0002",
            license = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            abstract = "Big abstract abacaba DABA-DABA",
            acmClass = "56asdf",
            reportNo = "IGPG-07/03-2"
        )

        val (arxivRecords, _, _) = ArxivXMLDomParser.parseArxivRecords(xmlText)

        assertTrue(arxivRecords.size == 1)
        assertEquals(expectedRecord, arxivRecords[0])
    }

    @Test
    fun testGetPdfLinks() {
        val xmlText = loadFromResourses("arxivApi.xml")
        val expectedUrlList = listOf("http://arxiv.org/pdf/1507.00493v3", "", "http://arxiv.org/pdf/1608.08082v6")

        val urlList = ArxivXMLDomParser.getPdfLinks(xmlText)

        assertEquals(expectedUrlList, urlList)
    }

    @Test
    fun testMakeOneLineOneWord() {
        val tstring = "  abacaba-  \ndaba-\n  caba  "
        val expected = "abacabadabacaba"

        val actual = ArxivXMLDomParser.makeOneLine(tstring)

        assertEquals(expected, actual)
    }

    @Test
    fun testMakeOneLineCompoundWord1() {
        val tstring = "  ABA-  \n   123  "
        val expected = "ABA-123"

        val actual = ArxivXMLDomParser.makeOneLine(tstring)

        assertEquals(expected, actual)
    }

    @Test
    fun testMakeOneLineCompoundWord2() {
        val tstring = "  ABA-  \n daba"
        val expected = "ABA-daba"

        val actual = ArxivXMLDomParser.makeOneLine(tstring)

        assertEquals(expected, actual)
    }

    @Test
    fun testMakeOneLineCompoundWord3() {
        val tstring = "  ABa-  \n Daba"
        val expected = "ABa-Daba"

        val actual = ArxivXMLDomParser.makeOneLine(tstring)

        assertEquals(expected, actual)
    }
}