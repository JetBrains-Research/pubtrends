package org.jetbrains.bio.pubtrends.pm

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

enum class PublicationType {
    ClinicalTrial,
    Dataset,
    TechnicalReport,
    Review,
    Article
}

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
    val auxInfo: ArticleAuxInfo = ArticleAuxInfo())