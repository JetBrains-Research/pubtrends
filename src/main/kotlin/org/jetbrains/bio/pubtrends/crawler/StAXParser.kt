package org.jetbrains.bio.pubtrends.crawler

import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.ABSTRACT_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.ARTICLE_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.AUTHOR_AFFILIATION_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.AUTHOR_INITIALS_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.AUTHOR_LASTNAME_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.AUTHOR_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.CITATION_PMID_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.DOI_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.JOURNAL_TITLE_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.KEYWORD_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.LANGUAGE_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.MEDLINE_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.OTHER_ABSTRACT_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.PMID_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.TITLE_TAG
import org.jetbrains.bio.pubtrends.crawler.PubmedXMLHandler.Companion.YEAR_TAG
import java.io.ByteArrayInputStream
import javax.xml.namespace.QName
import javax.xml.stream.XMLInputFactory
import javax.xml.stream.events.Characters

fun main(args: Array<String>) {
    val xmlString = """
        <A>
            Hello from A
            <B>
                Hello from B
            </B>
            A continues
        </A>
    """.trimIndent()

    val factory = XMLInputFactory.newFactory()
    val eventReader = factory.createXMLEventReader(ByteArrayInputStream(xmlString.toByteArray()))

    val articleList = mutableListOf<PubmedArticle>()

    // Temporary containers for information about articles and authors respectively
    var currentArticle = PubmedArticle()
    var currentAuthor = Author()

    var fullName = ""
    var isAbstractStructured = false

    while (eventReader.hasNext()) {
        val xmlEvent = eventReader.nextEvent()

        when {
            xmlEvent.isStartElement -> {
                val localName = xmlEvent.asStartElement().name.localPart
                println("$fullName: Start Element <$localName>")
                fullName = if (fullName.isEmpty()) localName else "$fullName/$localName"
            }
            xmlEvent.isCharacters -> {
                println("$fullName: Characters <${xmlEvent.asCharacters().data}>")
            }
            xmlEvent.isEndElement -> {
                val localName = xmlEvent.asEndElement().name.localPart
                println("$fullName: End Element <${localName}>")
                fullName = fullName.removeSuffix("/$localName")
            }
        }
    }
}