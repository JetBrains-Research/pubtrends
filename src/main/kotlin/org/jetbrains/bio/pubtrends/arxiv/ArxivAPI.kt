package org.jetbrains.bio.pubtrends.arxiv

import com.github.kittinunf.fuel.httpGet
import com.github.kittinunf.fuel.httpPost
import com.github.kittinunf.result.Result
import org.apache.logging.log4j.kotlin.logger
import org.jetbrains.bio.pubtrends.Config
import java.lang.Thread.sleep


/**
 * Used for accessing Arxiv bulk API and Arxiv API
 */
object ArxivAPI {

    private val logger = logger()
    const val requestBulkUrlPrefix = "http://export.arxiv.org/oai2?"
    const val requestApiUrlPrefix = "http://export.arxiv.org/api/query"

    //a time to sleep when receiving 503 code
    var sleepTime: Long = Config.config["arxiv_api_sleep_time"].toString().toLong()

    //a time to wait for a response
    var timeout = Config.config["arxiv_api_timeout"].toString().toInt()

    /**
     * Makes request to get metadata for all metadata from `startDate`
     * Each request returns only 1000 records and resumption token to use in the next request
     * If it is the first request to API, then resumption token should be null or empty,
     * otherwise resumption token received from the previous request should be given.
     * If `resumptionToken is not null and not empty, than `startDate` parameter is not used.
     * `limit` parameter is used to limit the number of records that are returned from each request
     *
     * Returns list of 1000 received records,
     * resumption token to use in the next request,
     * and the total number of records to be received
     */
    fun getBulkArxivRecords(
            startDate: String,
            resumptionToken: String? = null,
            limit: Int = 1000
    ): Triple<List<ArxivData>, String, Int> {

        logger.info("Begin api request from $startDate")
        logger.info("Using resumption token: $resumptionToken")

        val requestURL = when {
                resumptionToken.isNullOrEmpty() -> requestBulkUrlPrefix +
                            "verb=ListRecords&from=$startDate&metadataPrefix=arXiv"
                else -> requestBulkUrlPrefix +
                            "verb=ListRecords&resumptionToken=$resumptionToken"
        }
        val (_, response, result) = try {
            requestURL.httpGet().timeoutRead(timeout).responseString()
        } catch (e: Exception) {
            throw ApiRequestFailedException(e.message)
        }
        return when (result) {
            is Result.Failure -> {
                val ex = result.getException()
                if (response.statusCode == 503) {
                    logger.info("ArXiv OAI service is temporarily unavailable")
                    logger.info("Waiting ${sleepTime / 1000} seconds")
                    sleep(sleepTime)
                    getBulkArxivRecords(startDate, resumptionToken, limit)
                }
                else {
                    logger.error(ex)
                    logger.info("Failed: $ex")
                    throw ApiRequestFailedException(ex.message)
                }
            }

            is Result.Success -> {
                logger.info("Success")
                val data = result.get()

                val (arxivRecords, newResumptionToken, recordsTotal) = ArxivXMLDomParser.parseArxivRecords(data)
                if (resumptionToken == "") {
                    logger.info("Total records: $recordsTotal")
                }

                logger.info("Received ${arxivRecords.size} records")

                // TODO: disentangle
//                val pdfLinks = try {
//                    getRecordsUrl(arxivRecords.map { arxivData -> arxivData.id })
//                } catch (e : ApiRequestFailedException) {
//                    //try one more time
//                    getRecordsUrl(arxivRecords.map { arxivData -> arxivData.id })
//                }
//
//                for ((arxivData, pdfLink) in arxivRecords.zip(pdfLinks)) {
//                    arxivData.pdfUrl = pdfLink
//                }
                Triple(arxivRecords.take(limit), newResumptionToken, recordsTotal)
            }
        }
    }


    /**
     * Returns the list of pdf urls for each arxiv id from `idList`
     */
    fun getRecordsUrl(idList: List <String>): List<String> {
        val metadata = getArxivMetadata(idList)
        val records = ArxivXMLDomParser.getPdfLinks(metadata)
        if (records.size != idList.size) {
            throw ApiRequestFailedException(
                ("The number of records received from arxiv api(${records.size})" +
                        "differs from the number of ids(${idList.size})")
            )
        }
        return records
    }

    /**
     * Returns the list of full metadata about records with given ids
     */
    fun getArxivRecords(idList: List <String>): List<ArxivData> {
        val metadata = getArxivMetadata(idList)
        val records = ArxivXMLSaxParser.parse(metadata)
        if (records.size != idList.size) {
            throw ApiRequestFailedException(
                ("The number of records received from arxiv api(${records.size})" +
                        "differs from the number of ids(${idList.size})")
            )
        }
        records.forEachIndexed {i, record -> record.id = idList[i]}
        return records
    }

    /**
     * This function is used in `getRecordsUrl` and `getArxivRecords`
     * to make requests to arxiv API.
     * Returns the xml file received from API
     */
    private fun getArxivMetadata(idList: List <String>): String {
        logger.info("Begin api request to get arxiv metadata for ${idList.size} records")

        //form the string from ids
        val idString = idList.foldIndexed("") {i, acc, s ->
            if (i < idList.lastIndex)"$acc$s," else "$acc$s"
        }
        val (_, _, result) = requestApiUrlPrefix
            .httpPost(listOf("id_list" to idString, "max_results" to idList.size.toString()))
            .timeoutRead(timeout)
            .responseString()

        return when (result) {
            is Result.Failure -> {
                val ex = result.getException()
                logger.error("Failed: $ex")
                throw ApiRequestFailedException(ex.message)
            }
            is Result.Success -> {
                logger.info("Success: receive metadata")
                result.get()
            }
        }
    }

    class ApiRequestFailedException(message: String? = null) : Throwable(message)
}