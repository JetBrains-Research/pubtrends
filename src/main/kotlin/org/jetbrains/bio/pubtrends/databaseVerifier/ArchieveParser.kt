package org.jetbrains.bio.pubtrends.databaseVerifier

import com.google.gson.*
import java.util.*
import java.io.FileInputStream
import org.apache.logging.log4j.LogManager
import com.google.gson.reflect.TypeToken
import java.io.IOException


class ArchiveParser (
        private val archivePath: String
) {
    companion object {
        private val logger = LogManager.getLogger(ArchiveParser::class)
    }


    fun parse() {
        var inputStream: FileInputStream? = null
        var sc: Scanner? = null
        try {
            inputStream = FileInputStream(archivePath)
            sc = Scanner(inputStream, "UTF-8")
            val buffer = StringBuilder()
            var hashId: String?
            var pmid: Int? = 0
            var citations: List<String>?

            while (sc.hasNextLine()) {

                val line = sc.nextLine().trim()
                if (!line.endsWith("}")) {
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


                if (jsonObject != null) {
                    pmid = extractPmid(jsonObject.get("pmid"))
                }

                if (pmid != null) {
                    hashId = extractHashId(jsonObject?.get("id"))
                    citations = extractCitations(jsonObject?.get("outCitations"))
                    hashId?.let { citations?.let { citations -> addArticle(it, pmid, citations) } }
                }
            }
            if (sc.ioException() != null) {
                throw sc.ioException()
            }
        } catch (e: IOException) {
            logger.error("Please add to config file path to file, that has to be parsed")
        } finally {
            inputStream?.close()
            sc?.close()
        }
    }

    private fun extractHashId(idJson: JsonElement?): String? {
        var idString = idJson.toString()
        return deleteQuotes(idString)
    }

    private fun deleteQuotes(str : String): String {
        if (str.first() == '\"' && str.last() == '\"')
            return str.substring(1, str.length - 1)

        return str
    }


    private fun extractCitations(citationsJson: JsonElement?): List<String>? {
        val itemsType = object : TypeToken<List<String>>() {}.type
        return Gson().fromJson<List<String>>(citationsJson, itemsType).map { quoted -> deleteQuotes(quoted) }
    }

    private fun extractPmid(pmidJson: JsonElement): Int? {
        var pmidString = pmidJson.toString()
        val index = pmidString.indexOf("v") // version number is not needed
        val startIndex = 1
        if (index == -1) {
            if (pmidString.length > 2)
                return deleteQuotes(pmidString).toInt() // cut " "

            return null
        }

         return pmidString.substring(startIndex, index).toInt()
    }

    private fun lineToObject(line: String): JsonObject? {
        var jsonObject: JsonObject? = null
        try {
            jsonObject = JsonParser().parse(line).asJsonObject
        } catch (e: JsonSyntaxException) {
            logger.info("Parsing current article caused an error (probably something wrong with encoding), skipping article")
        }
        return jsonObject
    }

}