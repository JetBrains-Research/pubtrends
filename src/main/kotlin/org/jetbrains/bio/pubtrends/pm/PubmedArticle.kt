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

data class ArticleAuxInfo(
    val authors: List<Author> = listOf(),
    val databanks: List<DatabankEntry> = listOf(),
    val journal: Journal = Journal(),
    val language: String = ""
)

data class PubmedArticle(
    val pmid: Int = 0,
    val year: Int? = null,
    val title: String = "",
    val abstractText: String = "",
    val keywordList: List<String> = listOf(),
    val citationList: List<Int> = listOf(),
    val meshHeadingList: List<String> = listOf(),
    val type: PublicationType = PublicationType.Article,
    val doi: String = "",
    val auxInfo: ArticleAuxInfo = ArticleAuxInfo()
) {

    fun description(): Map<String, String> {
        return mapOf(
            "PMID" to pmid.toString(),
            "Year" to (year?.toString() ?: "undefined"),
            "Title" to title,
            "Type" to type.name,
            "Abstract Text" to abstractText,
            "DOI" to doi,
            "Keywords" to keywordList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
            "MesH" to meshHeadingList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
            "Citations" to citationList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
            "Other information" to auxInfo.toString()
        )
    }
}