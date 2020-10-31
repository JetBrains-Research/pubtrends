package org.jetbrains.bio.pubtrends.validation

import org.jetbrains.bio.pubtrends.algo.Algorithms
import org.jetbrains.bio.pubtrends.crossref.CRData
import org.jetbrains.bio.pubtrends.crossref.CrossRefApi
import org.jetbrains.bio.pubtrends.data.Reference
import org.apache.logging.log4j.kotlin.logger

object CrossRefValidator : Validator {
    val logger = logger()

    private val distThreshold = 0.05

    override fun validate(ref : Reference) {
        val records = CrossRefApi.findRecord(ref.rawReference)
        for (record in records) {
            if (checkSim(ref, record)) {
                ref.validated = true
                ref.title = record.title
                ref.doi = record.doi
                ref.firstPage = record.journal?.firstPage
                ref.lastPage = record.journal?.lastPage
                ref.issue = record.journal?.number
                ref.volume = record.journal?.volume
                ref.year = record.journal?.year
                ref.issn = record.journal?.issn
                ref.journal = record.journal?.shortTitle ?: record.journal?.fullTitle
                ref.authors = record.authors
                ref.urls.addAll(record.pdfUrls)
                break
            }
        }
    }

    private fun checkSim(ref : Reference, crRecord : CRData) : Boolean {
        val record = crDataToValidationRecord(crRecord)
        if (ref.title != null && ref.title != "" && !record.title.isNullOrEmpty()) {
            val dist = Algorithms.findLvnstnDist(ref.title!!, record.title)
            val d = dist.toDouble() / ref.title!!.length.toDouble()
            return d < distThreshold && SimilarityChecker.checkByTitle(ref, record)
        }
        else {
            return SimilarityChecker.check(ref, record)
        }
    }

    private fun crDataToValidationRecord(crRecord: CRData): ValidationRecord {
        return ValidationRecord(
            authors = crRecord.authors.map { ValidationRecord.Author(it.name) },
            journalName = if (crRecord.journal?.fullTitle.isNullOrBlank()) crRecord.journal?.shortTitle
                            else crRecord.journal?.fullTitle,
            journalVolume = crRecord.journal?.volume,
            firstPage = crRecord.journal?.firstPage,
            lastPage = crRecord.journal?.lastPage,
            title = crRecord.title,
            year = crRecord.journal?.year,
            issue = crRecord.journal?.number
        )
    }
}