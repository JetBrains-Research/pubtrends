package org.jetbrains.bio.pubtrends.ss

import com.google.gson.*
import com.google.gson.reflect.TypeToken
import org.apache.logging.log4j.LogManager
import java.io.File
import java.util.*
import java.util.zip.CRC32
import javax.xml.bind.DatatypeConverter


class ArchiveParser(
        private val archiveFile: File,
        private var batchSize: Int,
        private val addToDatabase: Boolean = true
) {
    val currentArticles: MutableList<SemanticScholarArticle> = mutableListOf()
    private var batchIndex = 0

    companion object {
        private val logger = LogManager.getLogger(ArchiveParser::class)
        private val crc32: CRC32 = CRC32()
    }


    fun parse() {
        var sc: Scanner
        archiveFile.inputStream().use {
            sc = Scanner(it, "UTF-8")
            val buffer = StringBuilder()
            var curArticle: SemanticScholarArticle
            while (sc.hasNextLine()) {
                val line = sc.nextLine().trim()

                if (!line.endsWith("}")) { // some articles are splitted into several lines
                    buffer.append(line)
                    continue
                }

                var jsonObject: JsonObject?
                if (buffer.isNotEmpty()) {
                    buffer.append(line)
                    jsonObject = lineToObject(buffer.toString())
                    buffer.clear()
                } else {
                    jsonObject = lineToObject(line)
                }

                if (jsonObject == null) {
                    continue
                }

                val ssid = jsonObject.get("id")?.asString ?: continue
                val title = jsonObject.get("title")?.asString ?: continue

                val pmid = extractPmid(jsonObject.get("pmid").asString)
                var doi = jsonObject.get("doi")?.asString
                if (doi == "") doi = null
                var abstract = jsonObject.get("paperAbstract")?.asString
                if (abstract == "") abstract = null
                var keywords: String? = jsonObject.get("entities").toString()
                        .replace("\"", "").removeSurrounding("[", "]")
                if (keywords == "") keywords = null

                var year: Int?
                if (jsonObject.get("year") == null) {
                    year = null
                } else {
                    year = jsonObject.get("year").asInt
                }

                val citationList = extractList(jsonObject.get("outCitations"))

                val authors = extractAuthors(jsonObject.get("authors"))
                val journal = extractJournal(jsonObject)
                val links = extractLinks(jsonObject)
                val venue = jsonObject.get("venue")?.asString ?: ""
                val aux = ArticleAuxInfo(authors, journal, links, venue)
                val source = getSource(journal, venue, links.pdfUrls)

                crc32.update(DatatypeConverter.parseHexBinary(ssid))
                val crc32id = crc32.value.toInt()
                crc32.reset()

                curArticle = SemanticScholarArticle(ssid = ssid, crc32id = crc32id, title = title,
                        pmid = pmid, doi = doi, abstract = abstract, keywords = keywords, year = year,
                        citationList = citationList, aux = aux, source = source)

                addArticleToBatch(curArticle)
            }
            handleEndDocument()
        }
    }

    private fun storeBatch() {
        DatabaseAdderUtils.addArticles(currentArticles)
        currentArticles.clear()
        batchIndex++
        logger.info("Finished batch $batchIndex adding ($archiveFile)")
    }

    private fun handleEndDocument() {
        if (currentArticles.isNotEmpty() && addToDatabase) {
            storeBatch()
        }
    }

    private fun addArticleToBatch(article: SemanticScholarArticle) {
        currentArticles.add(article)

        if (currentArticles.size == batchSize && addToDatabase) {
            storeBatch()
        }
    }

    private fun getSource(journal: Journal, venue: String, pdfUrls: List<String>): PublicationSource? {
        if (venue.equals("arxiv", ignoreCase = true) || pdfUrls.any { it.contains("arxiv.org", ignoreCase = true) }) {
            return PublicationSource.Arxiv
        }
        if (venue.equals("nature", ignoreCase = true) || journal.name.equals("nature", ignoreCase = true) ||
                pdfUrls.any { it.contains("nature.com", ignoreCase = true) }) {
            return PublicationSource.Nature
        }
        return null
    }

    private fun extractList(listJson: JsonElement?): List<String> {
        val itemsType = object : TypeToken<List<String>>() {}.type
        val list = Gson().fromJson<List<String>>(listJson, itemsType) ?: return listOf()
        return list.map { quoted -> quoted.removeSurrounding("\"") }
    }

    private fun extractPmid(pmidString: String): Int? {
        val index = pmidString.indexOf("v") // version number is not needed
        if (index == -1) {
            val pmid = pmidString.removeSurrounding("\"")
            if (pmid.isEmpty())
                return null

            return pmid.toInt()
        }

        return pmidString.substring(0, index).toInt()
    }

    private fun extractAuthors(authorsJson: JsonElement?): MutableList<Author> {
        val authors = authorsJson?.asJsonArray ?: return mutableListOf()
        val names = authors.map { author -> Author(author.asJsonObject.get("name").asString) }
        return names as MutableList<Author>
    }

    private fun extractLinks(articleJson: JsonObject): Links {
        val s2Url = articleJson.get("s2Url")?.asString
        val s2PdfUrl = articleJson.get("s2PdfUrl")?.asString
        val pdfUrls = extractList(articleJson.get("pdfUrls"))
        return Links(s2Url ?: "", s2PdfUrl ?: "", pdfUrls)
    }

    private fun extractJournal(articleJson: JsonObject): Journal {
        val name = articleJson.get("journalName")?.asString
        val volume = articleJson.get("journalVolume")?.asString
        val pages = articleJson.get("journalPages")?.asString
        return Journal(name ?: "", volume ?: "", pages ?: "")
    }


    private fun lineToObject(line: String): JsonObject? {
        var jsonObject: JsonObject? = null
        try {
            jsonObject = JsonParser().parse(line).asJsonObject
        } catch (e: JsonSyntaxException) {
            println(line)
            logger.info("Parsing current article caused an error (probably something wrong with encoding), skipping article")
        }

        return jsonObject
    }

}

