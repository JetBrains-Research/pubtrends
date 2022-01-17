package org.jetbrains.bio.pubtrends.ss

import com.google.gson.*
import com.google.gson.reflect.TypeToken
import org.jetbrains.bio.pubtrends.db.AbstractDBWriter
import org.slf4j.LoggerFactory
import java.io.File
import java.lang.Integer.min
import java.nio.file.Path
import java.util.*
import java.util.zip.GZIPInputStream


class SemanticScholarArchiveParser(
        private val dbWriter: AbstractDBWriter<SemanticScholarArticle>,
        private val archiveFileGz: File,
        private var batchSize: Int,
        private val collectStats: Boolean,
        private val statsTSV: Path
) {

    private val currentBatch = arrayListOf<SemanticScholarArticle>()
    private var batchIndex = 0

    companion object {
        private val LOG = LoggerFactory.getLogger(SemanticScholarArchiveParser::class.java)

        const val ARCHIVE_SIZE = 30000
    }

    init {
        if (collectStats) {
            LOG.info("Collecting stats in $statsTSV")
        }
    }

    // Stats about JSON tags
    private val tags = HashMap<String, Int>()

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

                val year: Int? = if (jsonObject.get("year") == null || jsonObject.get("year").toString() == "null") {
                    null
                } else {
                    try {
                        jsonObject.get("year").asInt
                    } catch (e: Exception) {
                        LOG.info("Skip year value: ${jsonObject.get("year")}")
                        null
                    }
                }

                val citationList = extractList(jsonObject.get("outCitations"))

                val authors = extractAuthors(jsonObject.get("authors"))
                val journal = extractJournal(jsonObject)
                val links = extractLinks(jsonObject)
                val venue = jsonObject.get("venue")?.asString ?: ""
                val aux = AuxInfo(authors, journal, links, venue)

                curArticle = SemanticScholarArticle(ssid = ssid, title = title,
                        pmid = pmid, doi = doi, abstract = abstract, keywords = keywords, year = year,
                        citations = citationList, aux = aux)

                addArticleToBatch(curArticle)

                if (collectStats) {
                    jsonObject.entrySet().forEach { e ->
                        tags[e.key] = (tags[e.key] ?: 0) + 1
                    }
                    for (i in citationList.indices) {
                        tags["citation"] = (tags["citation"] ?: 0) + 1
                    }
                    for (i in authors.indices) {
                        tags["author"] = (tags["author"] ?: 0) + 1
                    }
                }
            }
            handleEndDocument()
        }
    }

    private fun storeBatch() {
        dbWriter.store(currentBatch)
        currentBatch.clear()
        batchIndex++
        val progress = min(100, (1.0 * batchIndex * batchSize / ARCHIVE_SIZE * 100).toInt())
        LOG.info("Finished batch $batchIndex adding ($archiveFileGz) $progress%")
    }

    private fun handleEndDocument() {
        if (currentBatch.isNotEmpty()) {
            storeBatch()
        }
        if (collectStats) {
            LOG.info("Writing stats to $statsTSV")
            statsTSV.toFile().outputStream().bufferedWriter().use {
                it.write(archiveFileGz.path)
                it.newLine()
                tags.forEach { tag ->
                    it.write("${tag.key}\t${tag.value}\n")
                }
                it.newLine()
            }
        }
    }

    private fun addArticleToBatch(article: SemanticScholarArticle) {
        currentBatch.add(article)

        if (currentBatch.size == batchSize) {
            storeBatch()
        }
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
            LOG.error("Error parsing line, skipping article\n$line")
        }

        return jsonObject
    }

}

