package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.xml.sax.EntityResolver
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import java.io.File
import javax.xml.parsers.SAXParserFactory
import java.io.StringReader
import java.io.IOException


class PubmedXMLParser(val pubmedXMLHandler: PubmedXMLHandler) {
    companion object {
        private val logger = LogManager.getLogger(PubmedXMLParser::class)
    }

    private val spf = SAXParserFactory.newInstance()

    init {
        spf.isNamespaceAware = true
    }

    private val saxParser = spf.newSAXParser()
    private val xmlReader = saxParser.xmlReader

    init {
        xmlReader.contentHandler = pubmedXMLHandler
        xmlReader.entityResolver = NoDTDEntityResolver()
    }

    fun parse(name: String): Boolean {
        try {
            logger.debug("File location: ${File(name).absolutePath}")
            File(name).inputStream().use {
                xmlReader.parse(InputSource(it))
            }
        } catch (e: SAXException) {
            logger.error("Failed to parse $name", e)
        }

        return true
    }
}

class NoDTDEntityResolver : EntityResolver {
    /**
     * Overrides default EntityResolver to avoid downloading DTDs for every XML while parsing.
     */
    @Throws(SAXException::class, IOException::class)
    override fun resolveEntity(publicId: String?, systemId: String): InputSource? {
        return if (systemId.contains("dtd.nlm.nih.gov")) {
            InputSource(StringReader(""))
        } else {
            null
        }
    }
}