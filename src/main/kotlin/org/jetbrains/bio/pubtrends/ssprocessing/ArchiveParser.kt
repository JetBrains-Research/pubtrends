package org.jetbrains.bio.pubtrends.ssprocessing

import com.google.gson.*
import java.util.*
import org.apache.logging.log4j.LogManager
import com.google.gson.reflect.TypeToken
import java.io.*


class ArchiveParser(
        private val archiveFile: File,
        private var batchSize: Int,
        private val addCitations: Boolean,
        private val addToDatabase: Boolean = true
) {
    val currentArticles: MutableList<SemanticScholarArticle> = mutableListOf()
    private var batchIndex = 0

    companion object {
        private val logger = LogManager.getLogger(ArchiveParser::class)
    }


    fun parse() {
        var sc: Scanner
        archiveFile.inputStream().use {
            sc = Scanner(it, "UTF-8")
            val buffer = StringBuilder()
            var curArticle:SemanticScholarArticle
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
                curArticle = SemanticScholarArticle(ssid)
                curArticle.title = jsonObject.get("title")?.asString ?: continue

                if (!addCitations) {
                    curArticle.pmid = extractPmid(jsonObject.get("pmid"))
                    curArticle.doi = jsonObject.get("doi")?.asString
                    if (curArticle.doi == "") curArticle.doi = null
                    curArticle.abstract = jsonObject.get("paperAbstract")?.asString
                    if (curArticle.abstract == "") curArticle.abstract = null
                    curArticle.keywordList = extractList(jsonObject.get("entities")).filter {keyword -> keyword.length < 30}

                    if (jsonObject.get("year") == null) {
                        curArticle.year = null
                    } else {
                        curArticle.year = jsonObject.get("year").asInt
                    }

                    val authors = extractAuthors(jsonObject.get("authors"))
                    val journal = extractJournal(jsonObject)
                    val links = extractLinks(jsonObject)
                    val venue = jsonObject.get("venue")?.asString ?: ""
                    curArticle.aux = ArticleAuxInfo(authors, journal, links, venue)
                    curArticle.source = getSource(journal, venue, links.pdfUrls)

                } else {
                    curArticle.citationList = extractList(jsonObject.get("outCitations"))
                }
                addArticleToBatch(curArticle)
            }
            handleEndDocument()
        }
    }

    private fun storeBatch() {
        if (!addCitations) {
            DatabaseAdderUtils.addArticles(currentArticles)
        } else {
            DatabaseAdderUtils.addCitations(currentArticles)
        }
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
        if (venue.equals("arxiv", ignoreCase = true) || pdfUrls.any {it.contains("arxiv.org", ignoreCase = true)}) {
            return PublicationSource.Arxiv
        }
        if (venue.equals("nature", ignoreCase = true) || journal.name.equals("nature", ignoreCase = true) ||
                pdfUrls.any {it.contains("nature.com", ignoreCase = true)}) {
            return PublicationSource.Nature
        }
        return null
    }

    private fun deleteQuotes(str: String): String? {
        if (str.isEmpty()) return null
        if (str.first() == '\"' && str.last() == '\"')
            return str.substring(1, str.length - 1)

        return str
    }

    private fun extractList(listJson: JsonElement?): List<String> {
        val itemsType = object : TypeToken<List<String>>() {}.type
        val list = Gson().fromJson<List<String>>(listJson, itemsType) ?: return listOf()
        return list.mapNotNull { quoted -> deleteQuotes(quoted) }
    }

    private fun extractPmid(pmidJson: JsonElement?): Int? { //переделать
        var pmidString = pmidJson?.toString() ?: return null
        val index = pmidString.indexOf("v") // version number is not needed
        val startIndex = 1
        if (index == -1) {
            if (pmidString.length > 2)
                return deleteQuotes(pmidString)?.toInt()

            return null
        }

        return pmidString.substring(startIndex, index).toInt()
    }

    private fun extractAuthors(authorsJson: JsonElement?): MutableList<Author> {
        val authors = authorsJson?.asJsonArray ?:return mutableListOf()
        val names = authors.map{author ->  Author(author.asJsonObject.get("name").asString)}
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

