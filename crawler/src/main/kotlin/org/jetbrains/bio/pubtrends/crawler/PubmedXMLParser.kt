package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import java.io.File
import javax.xml.parsers.SAXParserFactory

class PubmedXMLParser(dbHandler : AbstractDBHandler) {
    companion object {
        private val logger = LogManager.getLogger(PubmedXMLParser::class)
    }

    private val spf = SAXParserFactory.newInstance()

    init {
        spf.isNamespaceAware = true
    }

    private val saxParser = spf.newSAXParser()
    internal val pubmedXMLHandler = PubmedXMLHandler(dbHandler)
    private val xmlReader = saxParser.xmlReader

    init {
        xmlReader.contentHandler = pubmedXMLHandler
    }

    fun parse(name : String) : Boolean {
        logger.info("$name: Parsing...")

        try {
            logger.debug("File location: ${File(name).absolutePath}")
            File(name).inputStream().use {
                xmlReader.parse(InputSource(it))
            }
        } catch (e: SAXException) {
            e.printStackTrace()
        }

        return true
    }
}