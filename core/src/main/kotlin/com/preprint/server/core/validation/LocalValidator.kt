package com.preprint.server.core.validation

import com.preprint.server.core.data.Author
import com.preprint.server.core.data.Reference
import com.preprint.server.validation.database.DBHandler
import com.preprint.server.validation.database.UniversalData
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.apache.logging.log4j.kotlin.logger

object LocalValidator : Validator, AutoCloseable {
    val logger = logger()
    val dbHandler =
        DBHandler(Config.config["validation_db_path"].toString())

    override fun validate(refList: List<Reference>) {
        fun mergeList(
            list1: List<List<UniversalData>>,
            list2: List<List<UniversalData>>
        ): List<List<UniversalData>> {
            return list1.zip(list2).map {(l1, l2) -> l1.union(l2).toList()}
        }

        fun toUdata(ref: Reference): UniversalData {
            return UniversalData(
                authors = ref.authors.map {UniversalData.Author(it.name)},
                title = ref.title,
                journalVolume = ref.volume,
                firstPage = ref.firstPage,
                lastPage = ref.lastPage,
                year = ref.year
            )
        }

        val urefList = refList.map {toUdata(it)}

        var titleData = listOf<List<UniversalData>>()
        var avpyData = listOf<List<UniversalData>>()
        var aflvData = listOf<List<UniversalData>>()
        var avyData = listOf<List<UniversalData>>()
        var apyData = listOf<List<UniversalData>>()

        runBlocking {

            launch {
                titleData = dbHandler.mgetByTitle(urefList)
            }
            launch {
                avpyData = dbHandler.mgetByAuthVolPageYear(urefList)
            }

            launch {
                aflvData = dbHandler.mgetByAuthFLPageVolume(urefList)
            }

            launch {
                avyData = dbHandler.mgetByAuthVolumeYear(urefList)
            }

            launch {
                apyData = dbHandler.mgetByAuthPageYear(urefList)
            }
        }

        val lists = listOf(titleData, avpyData, aflvData, avyData, apyData)
            .fold(List(titleData.size, { emptyList<UniversalData>()})) { acc, list ->
                mergeList(acc, list)
        }

        refList.zip(lists).forEach { (ref, list) -> bulkValidate(ref, list) }
    }

    fun bulkValidate(ref: Reference, list: List<UniversalData>) {
        list.forEach { undata ->
            if (check(ref, undata)) {
                accept(ref, undata)
                return
            }
        }
    }

    override fun validate(ref: Reference) {
        val records = mutableSetOf<UniversalData>()
        if (!ref.title.isNullOrBlank()) {
            records.addAll(dbHandler.getByTitle(ref.title!!))
            if (records.size == 1) {
                if (
                    ref.title!!.length > 20
                    && SimilarityChecker.checkByTitle(ref, universalToValidationRecord(records.first()))
                ) {
                    accept(ref, records.first())
                    return
                }
            }
        }

        if (!ref.volume.isNullOrBlank() && ref.firstPage != null && ref.authors.isNotEmpty()) {
            val auth = DBHandler.getFirstAuthorLetters(ref.authors.map {it.name})
            if (ref.year != null) {
                records.addAll(dbHandler.getByAuthVolPageYear(auth, ref.volume!!, ref.firstPage!!, ref.year!!))
            }
            if (ref.lastPage != null) {
                records.addAll(dbHandler.getByAuthFirsLastPageVolume(auth, ref.firstPage!!, ref.lastPage!!, ref.volume!!))
            }
        }


        if (ref.authors.size >= 2 && ref.year != null) {
            val authString = DBHandler.getFirstAuthorLetters(ref.authors.map {it.name})

            if (!ref.volume.isNullOrBlank()) {
                records.addAll(dbHandler.getByAuthorVolume(authString, ref.volume!!, ref.year!!))
            }

            if (ref.firstPage != null) {
                records.addAll(dbHandler.getByAuthorPage(authString, ref.firstPage!!, ref.year!!))
            }
        }


        records.forEach {
            if (check(ref, it)) {
                accept(ref, it)
                return
            }
        }
    }

    private fun check(ref: Reference, record: UniversalData): Boolean {
        return SimilarityChecker.check(ref, universalToValidationRecord(record))
    }

    private fun universalToValidationRecord(record: UniversalData): ValidationRecord {
        return ValidationRecord(
            authors = record.authors.map { ValidationRecord.Author(it.name)},
            journalName = record.journalName,
            journalVolume = record.journalVolume,
            title = record.title,
            year = record.year,
            firstPage = record.firstPage,
            lastPage = record.lastPage,
            issue = record.issue
        )
    }

    private fun accept(ref: Reference, record: UniversalData) {
        ref.title = record.title
        ref.authors = record.authors.map { Author(it.name) }
        ref.pmid = record.pmid
        ref.ssid = record.ssid
        ref.doi = record.doi
        ref.journal = record.journalName
        ref.firstPage = record.firstPage
        ref.lastPage = record.lastPage
        ref.volume = record.journalVolume
        ref.year = record.year
        ref.issue = record.issue
        ref.urls = record.pdfUrls
        ref.validated = true
        ref.isReference = true
    }

    override fun close() {
        dbHandler.close()
    }
}