package org.jetbrains.bio.pubtrends.databaseVerifier

import com.google.gson.*
import java.util.*
import java.io.FileInputStream
import org.apache.logging.log4j.LogManager
import com.google.gson.reflect.TypeToken
import java.io.File
import java.io.IOException


class ArchiveParser (
        private val archiveFile: File
) {
    companion object {
        private val logger = LogManager.getLogger(ArchiveParser::class)
    }


    fun parse() {
        var inputStream: FileInputStream? = null
        var sc: Scanner? = null
        try {
            inputStream = archiveFile.inputStream()
            sc = Scanner(inputStream, "UTF-8")
            var buffer = StringBuilder()
            var hashId: String?
            var pmid: Int? = 0
            var citations: List<String>?

            while (sc.hasNextLine()) {  // TODO: add articles using insertBatch instead of inserting each article separately
                val line = sc.nextLine().trim()
                if (!line.endsWith("}")) {
                    buffer.append(line)
                    continue
                }

                var jsonObject: JsonObject?
                if (buffer.isNotEmpty()) {
                    buffer.append(line)
                    jsonObject = lineToObject(buffer.toString())
                    buffer = StringBuilder()
                } else {
                    jsonObject = lineToObject(line)
                }


                if (jsonObject != null) {
                    pmid = extractPmid(jsonObject.get("pmid"))
                }

                if (pmid != null) {
                    hashId = extractHashId(jsonObject?.get("id"))
                    citations = extractCitations(jsonObject?.get("outCitations"))
                    hashId?.let { citations?.let { citations -> DatabaseAdderUtils().addArticle(it, pmid, citations) } }
                }
            }
            if (sc.ioException() != null) {
                throw sc.ioException()
            }
        } catch (e: IOException) {
            logger.error("File $archiveFile can't be parsed, please make sure that you added correct archive folder path to config.properties")
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
        val citationsList =  Gson().fromJson<List<String>>(citationsJson, itemsType)
        if (citationsList != null)
            return citationsList.map { quoted -> deleteQuotes(quoted) }

        return null
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