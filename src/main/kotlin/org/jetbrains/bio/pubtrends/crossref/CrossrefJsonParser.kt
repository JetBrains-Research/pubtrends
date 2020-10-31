package org.jetbrains.bio.pubtrends.crossref

import com.beust.klaxon.Klaxon
import org.jetbrains.bio.pubtrends.data.Author
import org.jetbrains.bio.pubtrends.data.JournalRef
import org.jetbrains.bio.pubtrends.utils.Common

/**
 * Used to parse data from CrossRef API response
 */
object CrossrefJsonParser {

    /**
     * Parses data from the CrossRef API response.
     * Parsed fields:
     * doi, title, authors, link to pdf(url),
     * and journal information(title(full and short), volume, pages, issue, issn, year).
     *
     * All parsed data is stored in CRData for each record in the response
     */
    fun parse(json: String): List<CRData> {
        val parsedJson = Klaxon().parse<CrossRefJsonData>(json)
        val items = parsedJson?.message?.items

        if (items != null) {
            val records = mutableListOf<CRData>()

            for (record in items) {
                val crRecord = CRData()
                record.DOI?.let {crRecord.doi = it}
                record.title?.let {crRecord.title = it[0]}

                crRecord.authors.addAll(record.author?.map {auth ->
                    Author(auth.family + " " + auth.given)
                } ?: listOf())

                record.link?.let {
                    crRecord.pdfUrls.addAll(it.map { it.URL }.filterNotNull())
                }

                //get journal information
                if (record.container_title != null) {
                    val journal = JournalRef("")
                    record.short_container_title?.let {journal.shortTitle = it[0]}
                    journal.fullTitle = record.container_title[0]
                    record.volume?.let {journal.volume = it}
                    record.page?.let {
                        val (firstPage, lastPage) = Common.splitPages(it)
                        journal.firstPage = firstPage
                        journal.lastPage = lastPage
                    }
                    record.issue?.let {journal.number = it}
                    record.ISSN?.let {journal.issn = it[0]}
                    val date = record.issued?.date_parts
                    if (!date.isNullOrEmpty() && !date[0].isNullOrEmpty() && date[0][0] != null) {
                        journal.year = date[0][0]
                    }
                    crRecord.journal = journal
                }
                records.add(crRecord)
            }
            return records
        }
        return listOf()
    }
}