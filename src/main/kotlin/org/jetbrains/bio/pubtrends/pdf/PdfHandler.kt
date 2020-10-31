package org.jetbrains.bio.pubtrends.pdf

import org.jetbrains.bio.pubtrends.data.PubData
import org.jetbrains.bio.pubtrends.ref.ReferenceExtractor

import com.github.kittinunf.fuel.httpGet
import com.github.kittinunf.result.Result
import org.jetbrains.bio.pubtrends.validation.Validator
import org.apache.logging.log4j.kotlin.logger
import org.jetbrains.bio.pubtrends.Config
import java.io.File
import java.lang.Exception
import java.lang.Thread.sleep


/**
 * Used to download pdf and extract references from them
 */
object PdfHandler {
    private val logger = logger()
    var sleepTime : Long = Config.config["arxiv_pdf_sleep_time"].toString().toLong()

    /**
     * Download pdf and extract references
     */
    fun getFullInfo(
        recordList: List <PubData>,
        outputPath: String,
        refExtractor: ReferenceExtractor?,
        validators: List<Validator>,
        savePdf: Boolean,
        totalRecordsToDownload: Int,
        beginRecordNumber: Int = 0
    ) {
        logger.info("Begin download of ${recordList.size} pdf")
        for ((i, record) in recordList.withIndex()) {
            logger.info("downloading ${beginRecordNumber + i + 1} out of ${totalRecordsToDownload}: ${record.id}")
            logger.info("pdf url: ${record.pdfUrl}")

            if (record.pdfUrl == "") {
                logger.error("Failed to download: pdf url is empty")
                File(outputPath + "failed.txt").appendText("${record.id}\n")
                continue
            }

            File(outputPath).mkdirs()
            // Load file from local storage or download if missing
            // Assuming that the link always looks like "http://arxiv.org/pdf/{record.id}v{version}"
            val pdfName = record.pdfUrl.split('/').last()
            val pdfFile = File("$outputPath${pdfName}.pdf")
            val pdf = try {
                if (pdfFile.exists()) {
                    pdfFile.readBytes()
                } else {
                    downloadPdf(record.pdfUrl) ?: continue
                }
            } catch (e: Exception) {
                logger.error("Failed to download: ${e.message}")
                continue
            }

            // Save if new file was downloaded and savePdf is true
            if (pdfFile.exists()) {
                logger.info("PDF file for record ${record.id} has been downloaded earlier")
            } else if (savePdf) {
                pdfFile.writeBytes(pdf)
            }

            if (refExtractor != null) {
                record.refList = try {
                    refExtractor.getReferences(pdf, validators).toMutableList()
                } catch (e: Exception) {
                    logger.error(e.stackTrace.asList().joinToString(separator = "\n"))
                    File(outputPath + "failed.txt").appendText("${record.id}\n")
                    continue
                }
                logger.debug(record.refList
                        .mapIndexed { i, ref -> "  ${i + 1}) $ref"}.joinToString(prefix = "\n", separator = "\n"))
            }

            sleep(sleepTime)
        }
    }

    fun downloadPdf(url : String) : ByteArray? {
        val (_, _, result) = url
            .httpGet()
            .response()
        return when (result) {
            is Result.Failure -> {
                val ex = result.getException()
                logger.error(ex)
                null
            }
            is Result.Success -> {
                logger.debug("Success: downloaded")
                result.get()
            }
        }
    }
}