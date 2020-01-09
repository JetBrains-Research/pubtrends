package org.jetbrains.bio.pubtrends.pm

import com.google.gson.Gson
import com.google.gson.GsonBuilder
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

data class ArticleAuxInfo(
    val authors: List<Author> = listOf(),
    val databanks: List<DatabankEntry> = listOf(),
    val journal: Journal = Journal(),
    val language: String = ""
)

data class PubmedArticle(
    val pmid: Int = 0,
    val date: DateTime? = null,
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
            "Year" to (date?.toString() ?: "undefined"),
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

    fun toNeo4j(): Map<String, String> {
        return mapOf(
                "pmid" to pmid.toString(),
                "title" to title.replace('\n', ' '),
                "abstract" to abstractText.replace('\n', ' '),
                "date" to (date?.toString() ?: ""),
                "type" to type.name,
                "aux" to GsonBuilder().create().toJson(auxInfo)
        )
    }
}