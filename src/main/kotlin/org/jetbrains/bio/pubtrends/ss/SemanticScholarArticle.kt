package org.jetbrains.bio.pubtrends.ss

import com.google.gson.GsonBuilder
import org.joda.time.DateTime

enum class PublicationSource {
    Nature,
    Arxiv
}

data class Author(var name: String = "")

data class Journal(var name: String = "", var volume: String = "", var pages: String = "")

data class Links(var s2Url: String = "", var s2PdfUrl: String = "", var pdfUrls: List<String> = listOf())

data class ArticleAuxInfo(val authors: List<Author> = listOf(),
                          val journal: Journal = Journal(),
                          val links: Links = Links(),
                          val venue: String = "")

// id:ID(SemanticScholar-ID)	pmid	date:date	title	abstract	type	keywords	doi	aux
data class SemanticScholarArticle(val ssid: String,
                                  val pmid: Int? = null,
                                  val citationList: List<String> = listOf(),
                                  val title: String = "",
                                  val abstract: String? = null,
                                  val year: Int? = null,
                                  val doi: String? = null,
                                  val keywords: String? = null,
                                  val source: PublicationSource? = null,
                                  val aux: ArticleAuxInfo = ArticleAuxInfo()) {

    fun toNeo4j(): Map<String, String?> {
        return mapOf(
                "ssid" to ssid,
                "pmid" to pmid?.toString(),
                "title" to title.replace('\n', ' '),
                "abstract" to abstract?.replace('\n', ' '),
                "date" to DateTime(year?:1970, 1, 1, 12, 0).toString(),
                "aux" to GsonBuilder().create().toJson(aux)
        )

    }
}


