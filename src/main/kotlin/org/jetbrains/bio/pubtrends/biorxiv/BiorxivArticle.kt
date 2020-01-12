package org.jetbrains.bio.pubtrends.biorxiv

import org.joda.time.LocalDate

data class BiorxivArticle(
        val biorxivId: String = "",
        val version: Int = 0,
        val date: LocalDate? = null,
        val title: String = "",
        val abstractText: String = "",
        val authors: List<String> = listOf(),
        val doi: String = "",
        val pdfUrl: String = ""
) {

    fun description(): Map<String, String> {
        return mapOf(
                "bioRxiv ID" to biorxivId,
                "Version" to version.toString(),
                "Date" to (date?.toString() ?: "undefined"),
                "Title" to title,
                "Abstract Text" to abstractText,
                "Authors" to authors.joinToString(", "),
                "DOI" to doi,
                "PDF URL" to pdfUrl
        )
    }

    // pmid:ID(Pubmed-ID)	date:date	title	abstract	type	keywords	mesh	doi	aux
    fun toNeo4j(): Map<String, String> {
        return mapOf(
                "biorxivId" to biorxivId,
                "version" to version.toString(),
                "title" to title.replace('\n', ' '),
                "abstract" to abstractText.replace('\n', ' '),
                "date" to (date?.toString() ?: ""),
                "doi" to doi,
                "pdfUrl" to pdfUrl
        )
    }
}