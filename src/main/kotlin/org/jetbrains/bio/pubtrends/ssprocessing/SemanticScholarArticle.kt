package org.jetbrains.bio.pubtrends.ssprocessing


data class Author(var name: String = "")


data class Journal(var name: String = "", var volume: String = "", var pages: String = "")


data class Links(var s2Url: String = "", var s2PdfUrl: String = "", var pdfUrls: List<String> = mutableListOf())


data class ArticleAuxInfo(val authors: MutableList<Author> = mutableListOf(),
                          val journal: Journal = Journal(),
                          val links: Links = Links(),
                          val venue: String = "")


data class SemanticScholarArticle(var ssid: String) {
    var pmid: Int? = null
    var citationList: List<String> = listOf()
    var title: String = ""
    var abstract: String? = null
    var year: Int? = null
    var doi: String? = null
    var keywordList: List<String> = mutableListOf()
    var source: PublicationSource? = null
    var aux: ArticleAuxInfo = ArticleAuxInfo()
    var id: Int = 0
}
