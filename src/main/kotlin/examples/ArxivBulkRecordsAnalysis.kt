package examples

import org.jetbrains.bio.pubtrends.arxiv.ArxivAPI
import org.jetbrains.bio.pubtrends.arxiv.ArxivAPI.getRecordsUrl
import org.jetbrains.bio.pubtrends.arxiv.ArxivPDFHandler
import org.jetbrains.bio.pubtrends.ref.GrobidReferenceExtractor
import java.time.LocalDate

fun main() {
    val startDate = "2020-10-29"
    val requestDate = LocalDate.parse(startDate)
    val (records, token, recordsTotal) = ArxivAPI.getBulkArxivRecords(startDate, limit = 100)

    println("Retrieved $recordsTotal papers")
    println("Resumption token: $token")

    val newRecords = records.filter { record ->
        val date = record.lastUpdateDate.let { LocalDate.parse(it) }
        return@filter date >= requestDate
    }
    println("Found ${newRecords.size} new papers after $startDate")

    val pdfLinks = try {
        getRecordsUrl(newRecords.map { arxivData -> arxivData.id })
    } catch (e: ArxivAPI.ApiRequestFailedException) {
        //try one more time
        getRecordsUrl(newRecords.map { arxivData -> arxivData.id })
    }

    for ((arxivData, pdfLink) in newRecords.zip(pdfLinks)) {
        arxivData.pdfUrl = pdfLink
    }

    ArxivPDFHandler.getFullInfo(
            newRecords,
            "local/pdf/",
            GrobidReferenceExtractor,
            listOf(),
            true,
            newRecords.size,
            0
    )

    newRecords.forEach {
        println("*** RECORD ${it.identifier} ***")
        println("PDF link: ${it.pdfUrl}")
        println("References:")
        it.refList.forEach { ref -> println(ref.rawReference) }
    }
}