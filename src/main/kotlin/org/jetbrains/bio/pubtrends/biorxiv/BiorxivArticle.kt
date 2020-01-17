package org.jetbrains.bio.pubtrends.biorxiv

import org.joda.time.LocalDate

data class BiorxivArticle(
        val biorxivId: Int = 0,
        val version: Int = 0,
        val date: LocalDate? = null,
        val title: String = "",
        val abstractText: String = "",
        val authors: List<String> = listOf(),
        val doi: String = "",
        val pdfUrl: String = ""
) {
    fun toNeo4j(): Map<String, String> {
        return mapOf(
                "biorxivId" to biorxivId.toString(),
                "version" to version.toString(),
                "title" to title.replace('\n', ' '),
                "abstract" to abstractText.replace('\n', ' '),
                "date" to (date?.toString() ?: ""),
                "doi" to doi,
                "pdfUrl" to pdfUrl
        )
    }
}