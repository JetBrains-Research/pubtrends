package com.preprint.server.core.arxiv

import com.preprint.server.core.neo4j.DatabaseHandler
import com.preprint.server.core.pdf.PdfHandler
import com.preprint.server.core.ref.CustomReferenceExtractor
import com.preprint.server.core.validation.Validator

import org.apache.logging.log4j.kotlin.logger
import java.lang.Thread.sleep
import java.time.LocalDate
import java.time.temporal.ChronoUnit


/**
 * Used to collect data about arxiv publications
 */
object ArxivCollector {
    private val logger = logger()
    var resumptionToken = ""


    //the time to sleep when arxiv API request fails
    var sleepTime: Long = Config.config["arxiv_api_sleep_time"].toString().toLong()


    /**
     * Collects all data sequentially from the `startDate`,
     * get references for all received records
     * (downloads pdf for each record and extracts references),
     * and then stores all received data into the database
     *
     * If `resumptionToken` is empty then makes request with the given,
     * otherwise make requests with given resumption token to the arxiv api
     * and 'startDate` will be ignored by the ArxivApi object later
     * (read more about working with resumption token in ArxivApi description)
     *
     * limit is the number of records to get from each request
     * (arxiv api response contains 1000 records)
     */
    fun collect(
        startDate: String,
        dbHandler: DatabaseHandler,
        validators: List<Validator>,
        resumptionToken_: String = "",
        limit: Int = 1000
    ) {
        var recordsProcessed = 0
        resumptionToken = resumptionToken_

        if (resumptionToken_ == "") {
            logger.info("Begin collecting arxiv metadata from $startDate")
        } else {
            logger.info("Continue collecting arxiv metadata from $startDate with resumption token:$resumptionToken")
        }

        //do request until resumption token in the response will be empty,
        //that means that this was the last pack of records
        do {
            val (newArxivRecords, newResumptionToken, recordsTotal) = try {
                ArxivAPI.getBulkArxivRecords(startDate, resumptionToken, limit)
            }  catch (e: ArxivAPI.ApiRequestFailedException) {
                sleep(sleepTime)
                continue
            }
            resumptionToken = newResumptionToken

            val newUpdatedRecords = newArxivRecords.filter {record ->
                val date = record.lastUpdateDate?.let {LocalDate.parse(it)} ?: LocalDate.parse(record.creationDate)
                val requestDate = LocalDate.parse(startDate)
                if (requestDate > date) {
                    val days = ChronoUnit.DAYS.between(date, requestDate)
                    days < 7
                } else {
                    true
                }
            }

            //get references for all records, and store them
            // in the `refList` property of each record in `newArxivRecords`
            PdfHandler.getFullInfo(
                newUpdatedRecords,
                "files/",
                CustomReferenceExtractor,
                validators,
                false,
                recordsTotal,
                recordsProcessed
            )

            dbHandler.storeArxivData(newArxivRecords)

            recordsProcessed += newArxivRecords.size
            logger.info("Records processed ${recordsProcessed} out of $recordsTotal")
        } while (resumptionToken != "")
    }
}