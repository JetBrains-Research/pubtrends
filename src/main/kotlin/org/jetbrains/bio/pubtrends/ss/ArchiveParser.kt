package org.jetbrains.bio.pubtrends.ss

import com.google.gson.*
import com.google.gson.reflect.TypeToken
import org.apache.commons.codec.binary.Hex
import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.batchInsert
import org.jetbrains.exposed.sql.transactions.transaction
import java.io.File
import java.util.*
import java.util.zip.CRC32
import java.util.zip.GZIPInputStream


class ArchiveParser(
        private val archiveFileGz: File,
        private var batchSize: Int,
        private val addToDatabase: Boolean = true,
        private val curFile: Int,
        private val filesAmount: Int
) {
    val currentArticles: MutableList<SemanticScholarArticle> = mutableListOf()
    private var batchIndex = 0

    companion object {
        private val logger = LogManager.getLogger(ArchiveParser::class)
        private val crc32: CRC32 = CRC32()
    }


    fun parse() {
        var sc: Scanner
        GZIPInputStream(archiveFileGz.inputStream()).use {
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

                curArticle = SemanticScholarArticle(ssid = ssid, title = title,
                        pmid = pmid, doi = doi, abstract = abstract, keywords = keywords, year = year,
                        citationList = citationList, aux = aux, source = source)

                addArticleToBatch(curArticle)
            }
            handleEndDocument()
        }
    }

    private fun storeBatch() {
        addArticles(currentArticles)
        currentArticles.clear()
        batchIndex++
        val progress = batchIndex.toDouble() / 10 //number of batches is about 1000
        logger.info("Finished batch $batchIndex adding ($archiveFileGz) ($progress% done of $curFile/$filesAmount file)")
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

    private fun addArticles(articles: List<SemanticScholarArticle>) {
        val citationsList = articles.map { it.citationList.distinct().map { cit -> it.ssid to cit } }.flatten()

        transaction {
            SSPublications.batchInsert(articles, ignore = true) { article ->
                this[SSPublications.ssid] = article.ssid
                this[SSPublications.pmid] = article.pmid
                this[SSPublications.abstract] = article.abstract
                this[SSPublications.keywords] = article.keywords
                this[SSPublications.title] = article.title
                this[SSPublications.year] = article.year
                this[SSPublications.doi] = article.doi
                this[SSPublications.sourceEnum] = article.source
                this[SSPublications.aux] = article.aux
                this[SSPublications.crc32id] = crc32id(article.ssid)
            }

            SSCitations.batchInsert(citationsList, ignore = true) { citation ->
                this[SSCitations.id_out] = citation.first
                this[SSCitations.id_in] = citation.second
                this[SSCitations.crc32id_out] = crc32id(citation.first)
                this[SSCitations.crc32id_in] = crc32id(citation.second)
            }
        }
    }

    private fun crc32id(ssid: String): Int {
        crc32.reset()
        crc32.update(Hex.decodeHex(ssid.toCharArray()))
        return crc32.value.toInt()
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
        val pmidWithoutVersion = pmidString.substringBefore("v")
        if (pmidWithoutVersion.isEmpty())
            return null

        return pmidWithoutVersion.toInt()
    }

    private fun extractAuthors(authorsJson: JsonElement?): List<Author> {
        val authors = authorsJson?.asJsonArray ?: return listOf()
        return authors.map { author -> Author(author.asJsonObject.get("name").asString) }
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

