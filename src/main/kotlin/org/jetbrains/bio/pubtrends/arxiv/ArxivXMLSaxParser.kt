package org.jetbrains.bio.pubtrends.arxiv

import org.jetbrains.bio.pubtrends.data.Author
import org.jetbrains.bio.pubtrends.data.JournalRef
import org.xml.sax.Attributes
import org.xml.sax.helpers.DefaultHandler
import java.io.ByteArrayInputStream
import javax.xml.parsers.SAXParserFactory


/**
 * SAX parser that is used to parse data from arxiv API xml responses
 * Currently DOM and SAX parsers is used to parse different kinds of responses
 */
object ArxivXMLSaxParser {
    private val factory = SAXParserFactory.newInstance()
    private val parser = factory.newSAXParser()

    /**
     * Parses full metadata from the arxiv api response about multiple records
     * and saves it in the ArxivData class for each record
     */
    fun parse(xmlText: String): List<ArxivData> {
        val handler = ArxivHandler()
        parser.parse(ByteArrayInputStream(xmlText.toByteArray()), handler)
        return handler.records
    }

    /**
     * This handler is used in SAX-parser
     */
    private class ArxivHandler : DefaultHandler() {
        val records = mutableListOf<ArxivData>()

        //stores information about tags, `tagStatus["tag"]` == 1 means that the tag is open
        private val tagStatus = mutableMapOf<String, Boolean>().withDefault { false }

        private var curRecord = ArxivData()

        /*
        SAX parser parses only one line at a time(and maybe sometimes can split one line in two)
        and arxiv api response contains multiline text, so we save lines for each tag in this map:
         */
        private val lines = mutableMapOf<String, MutableList<String>>()

        override fun startDocument() {
            curRecord = ArxivData()
            lines.clear()
            tagStatus.clear()
            records.clear()
        }

        override fun endDocument() {
            curRecord = ArxivData()
            lines.clear()
            tagStatus.clear()
        }

        override fun startElement(uri: String?, localName: String?, qName: String?, attributes: Attributes?) {
            if (qName == null) {
                return
            }
            when (qName) {
                "entry"     -> curRecord = ArxivData()
                "category"  -> {
                    if (attributes != null) {
                        val cat = attributes.getValue("term")
                        if (!cat.isNullOrBlank() && !cat.contentEquals(" ") && cat.all {!it.isDigit()}) {
                            curRecord.categories.add(cat)
                        }
                    }
                }
                "link"     -> {
                    if (attributes != null && attributes.getValue("title") == "pdf") {
                        val link = attributes.getValue("href")
                        if (!link.isNullOrBlank()) {
                            curRecord.pdfUrl = link
                        }
                    }
                }
                else       -> {
                    tagStatus[qName] = true
                    lines[qName] = mutableListOf()
                }
            }
        }

        override fun endElement(uri: String?, localName: String?, qName: String?) {
            if (qName == null) {
                return
            }
            tagStatus[qName] = false
            when (qName) {
                "entry"     -> records.add(curRecord)
                "title"     -> curRecord.title = getFullText("title")!!
                "summary"   -> curRecord.abstract = getFullText("summary")!!
                "published" -> curRecord.creationDate = getFullString("published")!!.substringBefore("T")
                "updated"   -> curRecord.lastUpdateDate = (getFullString("updated") ?: curRecord.creationDate).substringBefore("T")
                "author"    -> {
                                    val aff = getFullString("arxiv:affiliation")
                                    curRecord.authors.add(Author(getFullString("name")!!, aff))
                               }
                "arxiv:doi" -> curRecord.doi = getFullString("arxiv:doi")
                "arxiv:journal_ref" -> getFullString("arxiv:journal_ref")?.let {
                                   curRecord.journal = JournalRef(it, true)
                               }
            }
        }

        @ExperimentalStdlibApi
        override fun characters(ch: CharArray?, start: Int, length: Int) {
            if (ch == null) {
                return
            }
            val value = getValue(ch, start, length)
            if (tagStatus.getValue("published")) {
                lines["published"]!!.add(value)
            }
            if (tagStatus.getValue("updated")) {
                lines["updated"]!!.add(value)
            }
            if (tagStatus.getValue("title")) {
                lines["title"]!!.add(value)
            }
            if (tagStatus.getValue("summary")) {
                lines["summary"]!!.add(value)
            }
            if (tagStatus.getValue("author") && tagStatus.getValue("name")) {
                lines["name"]!!.add(value)
            }
            if (tagStatus.getValue("author") && tagStatus.getValue("arxiv:affiliation")) {
                lines["arxiv:affiliation"]!!.add(value)
            }
            if (tagStatus.getValue("arxiv:doi")) {
                lines["arxiv:doi"]!!.add(value)

            }
            if (tagStatus.getValue("arxiv:journal_ref")) {
                lines["arxiv:journal_ref"]!!.add(value)
            }
        }

        private fun getValue(ch: CharArray, start: Int, length: Int): String {
            return String(ch, start, length)
        }

        /**
         * Merge all string for `tag` into one line string(for text fields)
         */
        private fun getFullText(tagName: String): String? {
            return makeOneLine(lines[tagName]?.toList() ?: return null)
        }

        /**
         * Merge all string for `tag` into one line string(for non text fields)
         */
        private fun getFullString(tagName: String): String? {
            return makeOneLineNotText(lines[tagName]?.toList() ?: return null)
        }

        /**
         * Converts multiline string into one line string
         */
        private fun makeOneLine(rawLines: List<String>): String {
            val lines = rawLines.map {it.trim()}.filter { it.isNotEmpty() }
            var res = ""
            for ((i, line) in lines.withIndex()) {
                if (i == 0) {
                    res = line
                    continue
                }
                if (res.length > 1 && line.length > 0 && res.last() == '-') {
                    val lastC = res[res.lastIndex - 1]
                    if (lastC.isLowerCase() && line.first().isLowerCase()) {
                        res = res.dropLast(1) + line
                    }
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

        private fun makeOneLineNotText(rawLines: List<String>): String {
            return rawLines.joinToString(separator = "")
        }
    }
}