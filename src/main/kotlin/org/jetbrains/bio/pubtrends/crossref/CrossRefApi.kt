package com.preprint.server.core.crossref

import com.github.kittinunf.fuel.core.Response
import com.github.kittinunf.fuel.httpGet
import com.github.kittinunf.result.Result
import com.preprint.server.core.utils.RequestLimiter
import org.apache.logging.log4j.kotlin.logger
import java.lang.Thread.sleep
import java.net.URLEncoder


/**
 * Used for accessing CrossRef API
 */
object CrossRefApi {
    private val logger = logger()
    const val prefix = "https://api.crossref.org"

    val email = Config.config["email"].toString()

    //the number of records that each request returns
    var maxRecordsNumber = Config.config["crossref_max_records_returned"].toString()

    private val defaultRequestLimit = Config.config["crossref_request_max"].toString().toInt()
    private val defaultRequestInterval = Config.config["crossref_request_interval"].toString().toLong()
    private val reqLimiter = RequestLimiter(
        defaultRequestLimit,
        defaultRequestInterval
    )

    private val timeout = Config.config["crossref_timeout"].toString().toInt()

    private val testTimeout = Config.config["crossref_test_timeout"].toString().toInt()

    private val checkConnectioRequest = Config.config["crossref_test_request"].toString()

    private val sleepTime = Config.config["crossref_sleep_time"].toString().toLong()


    /**
     * Makes request to CrossRef API to find the given record
     * and returns `maxRecordsNumber` most suitable results
     */
    fun findRecord(ref: String): List<CRData> {

        //if there was made too many request, will wait until can make another one
        reqLimiter.waitForRequest()

        val url = "$prefix/works?query=${URLEncoder.encode(ref, "utf-8")}&rows=$maxRecordsNumber&mailto=$email"
        val (_, response, result) = try {
            url.httpGet().timeoutRead(timeout).responseString()
        } catch (e : Exception) {
            waitConnection()
            throw ApiRequestFailedException(e.message)
        }
        val (newLimit, newInterval) = getNewInterval(response)
        reqLimiter.set(newLimit, newInterval)

        when (result) {
            is Result.Failure -> {
                val ex = result.getException()
                if (response.statusCode == 414) {
                    //this means that url is too long, and most likely
                    //reference was parsed wrong
                    return listOf()
                }
                else {
                    waitConnection()
                    throw ApiRequestFailedException(ex.message)
                }
            }
            is Result.Success -> {
                val records = CrossrefJsonParser.parse(result.value)
                return records
            }
        }
    }

    private fun getNewInterval(response: Response): Pair<Int, Long> {
        val newLimit =  response.headers.get("X-Rate-Limit-Limit").toList()
        val newInterval = response.headers.get("X-Rate-Limit-Interval").toList()
        if (newLimit.isEmpty() || newInterval.isEmpty()) {
            return Pair(50, 2100.toLong())
        }
        else {
            return Pair(newLimit[0].toInt() - 1, newInterval[0].dropLast(1).toLong() * 1000 * 2 + 100)
        }
    }

    private fun waitConnection() {
        while (true) {
            val (_, _, result) = try {
                checkConnectioRequest.httpGet().timeoutRead(testTimeout).responseString()
            } catch (e: Exception) {
                sleep(sleepTime)
                continue
            }
            if (result is Result.Failure) {
                logger.error(result.getException())
                logger.error("Waiting for CrossRef to be available")
                sleep(sleepTime)
                continue
            }
            break
        }
    }

    class ApiRequestFailedException(override val message: String?) : Exception(message)
}