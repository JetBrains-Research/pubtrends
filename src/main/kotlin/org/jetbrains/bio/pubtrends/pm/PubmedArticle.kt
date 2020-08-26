package org.jetbrains.bio.pubtrends.pm

import org.joda.time.DateTime

data class Author(
        val name: String = "",
        val affiliation: List<String> = listOf()
)

data class Journal(val name: String = "")

data class DatabankEntry(
        val name: String = "",
        val accessionNumber: List<String> = listOf()
)

data class Aux(
        val authors: List<Author> = listOf(),
        val databanks: List<DatabankEntry> = listOf(),
        val journal: Journal = Journal(),
        val language: String = ""
)

enum class PublicationType {
    ClinicalTrial,
    Dataset,
    TechnicalReport,
    Review,
    Article
}

data class PubmedArticle(
        val pmid: Int = 0,
        val title: String = "",
        val abstract: String = "",
        val date: DateTime? = null,
        val type: PublicationType = PublicationType.Article,
        val keywords: List<String> = listOf(),
        val mesh: List<String> = listOf(),
        val doi: String = "",
        val aux: Aux = Aux(),
        val citations: List<Int> = listOf()
)