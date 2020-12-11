package org.jetbrains.bio.pubtrends.pm

data class Author(
        val name: String = "",
        val affiliation: List<String> = listOf()
)

data class Journal(val name: String = "")

data class DatabankEntry(
        val name: String = "",
        val accessionNumber: List<String> = listOf()
)

data class AuxInfo(
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
        val year: Int = 1970,
        val type: PublicationType = PublicationType.Article,
        val keywords: List<String> = listOf(),
        val mesh: List<String> = listOf(),
        val doi: String = "",
        val aux: AuxInfo = AuxInfo(),
        val citations: List<Int> = listOf()
)