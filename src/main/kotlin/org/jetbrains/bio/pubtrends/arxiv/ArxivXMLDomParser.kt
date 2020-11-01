package org.jetbrains.bio.pubtrends.arxiv

import org.jetbrains.bio.pubtrends.data.Author
import org.jetbrains.bio.pubtrends.data.JournalRef
import org.w3c.dom.Element
import org.xml.sax.InputSource
import java.io.ByteArrayInputStream
import javax.xml.parsers.DocumentBuilderFactory


/**
 * DOM parser that is used to parse data from arxiv API xml responses
 * Currently DOM and SAX parsers is used to parse different kinds of responses
 */
object ArxivXMLDomParser {
    /**
     * Parses data from the response of the arxiv bulk API response
     * Returns parsed data, resumption token, and total number of records
     * that we will receive information about
     *
     * Fields that must always be presented:
     * identifier, datestamp, id, title, abstract, creation date
     * otherwise the record will be ignored and won't be added to the list
     */
    fun parseArxivRecords(xmlText: String): Triple<List<ArxivData>, String, Int> {
        val inputStream = InputSource(ByteArrayInputStream(xmlText.toByteArray()))
        val xmlDoc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(inputStream)
        xmlDoc.documentElement.normalize()

        val arxivRecords = mutableListOf<ArxivData>()
        val recordList = xmlDoc.getElementsByTagName("record")
        for (i in 0 until recordList.length) {
            val recordElem = recordList.item(i) as Element
            val recordHeader = recordElem.getElementsByTagName("header").item(0) as Element
            val recordMetadata = recordElem.getElementsByTagName("metadata").item(0) as Element
            recordHeader.normalize()
            recordMetadata.normalize()

            // create ArxivData from identifier string
            val arxivData = ArxivData(
                recordHeader.getValue("identifier") ?: continue
            )

            // Required fields
            arxivData.datestamp = recordHeader.getValue("datestamp") ?: continue
            arxivData.id = recordMetadata.getValue("id") ?: continue
            arxivData.creationDate = recordMetadata.getValue("created") ?: continue
            arxivData.title = recordMetadata.getValue("title") ?: continue
            arxivData.abstract = recordMetadata.getValue("abstract") ?: continue

            // Optional fields
            arxivData.lastUpdateDate = recordMetadata.getValue("updated") ?: arxivData.creationDate

            // get authors' names with affiliations(if present)
            val authorsNodeList = recordMetadata.getElementsByTagName("authors").item(0) as Element
            val authorsList = authorsNodeList.getElementsByTagName("author")
            for (j in 0 until authorsList.length) {
                val authorInfo = authorsList.item(j) as Element
                var name = authorInfo.getValue("keyname") ?: ""
                val forenames = authorInfo.getValue("forenames")
                val suffix = authorInfo.getValue("suffix")
                if (forenames != null) {
                    name = "$forenames $name"
                }
                if (suffix != null) {
                    name = "$name $suffix"
                }

                val affiliation : String? = authorInfo.getValue("affiliation")
                arxivData.authors.add(Author(name, affiliation))
            }

            arxivData.categories = recordMetadata.getValue("categories")
                ?.split(" ")?.toMutableList() ?: mutableListOf()

            recordMetadata.getValue("journal-ref")?.let {
                // FIXME: disabled parsing in order to disentangle reference processing
                arxivData.journal = JournalRef(it, false)
            }
            arxivData.comments = recordMetadata.getValue("comments")
            arxivData.reportNo = recordMetadata.getValue("report-no")
            arxivData.mscClass = recordMetadata.getValue("msc-class")
            arxivData.acmClass = recordMetadata.getValue("acm-class")
            arxivData.doi = recordMetadata.getValue("doi")
            arxivData.license = recordMetadata.getValue("license")
            arxivRecords.add(arxivData)
        }

        val resumptionTokenElems = xmlDoc.getElementsByTagName("resumptionToken")
        var recordsTotal = 0
        var resumptionToken = ""
        if (resumptionTokenElems.length != 0) {
            val resumptionTokenElem = resumptionTokenElems.item(0) as Element
            recordsTotal = resumptionTokenElem.getAttribute("completeListSize").toInt()
            resumptionToken = resumptionTokenElem.textContent
        }

        return Triple(arxivRecords, resumptionToken, recordsTotal)
    }

    /**
     * Parses only pdf urls from the arxiv api response for multiple records
     */
    fun getPdfLinks(xmlText: String) : List<String> {
        val pdfList = mutableListOf<String>()

        val inputStream = InputSource(ByteArrayInputStream(xmlText.toByteArray()))
        val xmlDoc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(inputStream)

        val entryList = xmlDoc.getElementsByTagName("entry")
        for (i in 0 until entryList.length) {
            val elem = entryList.item(i) as Element
            val links = elem.getElementsByTagName("link")
            var added = false
            for (j in 0 until links.length) {
                val linkElem = links.item(j) as Element
                if (linkElem.hasAttribute("title") && linkElem.getAttribute("title") == "pdf") {
                    pdfList.add(linkElem.getAttribute("href"))
                    added = true
                }
            }
            if (!added) {
                pdfList.add("")
            }
        }
        return pdfList
    }

    private fun Element.getValue(tagName : String) : String? {
        val elems =  this.getElementsByTagName(tagName)
        if (elems.length == 0) {
            return null
        }
        else {
            return makeOneLine(elems.item(0).textContent)
        }
    }

    /**
     * Converts multiline string(with '\n') to one line string
     * Examines cases when line ends with '-', which can mean that
     * this word continues in the next string
     */
    fun makeOneLine(str: String): String {
        val lines = str.split("\n").map {it.trim()}.filter { it.isNotEmpty() }
        var res = ""
        for ((i, line) in lines.withIndex()) {
            if (i == 0) {
                res = line
                continue
            }
            if (res.length > 1 && line.length > 0 && res.last() == '-') {
                val lastC = res[res.lastIndex - 1]
                if (lastC.isLowerCase() && line.first().isLowerCase()) res = res.dropLast(1) + line
                else {
                    res += line
                }
            }
            else {
                res += " " + line
            }
        }
        return res
    }
}