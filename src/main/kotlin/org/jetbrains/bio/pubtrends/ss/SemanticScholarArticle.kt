package org.jetbrains.bio.pubtrends.ss


data class Author(var name: String = "")


data class Journal(var name: String = "", var volume: String = "", var pages: String = "")


data class Links(var s2Url: String = "", var s2PdfUrl: String = "", var pdfUrls: List<String> = mutableListOf())


data class ArticleAuxInfo(val authors: MutableList<Author> = mutableListOf(),
                          val journal: Journal = Journal(),
                          val links: Links = Links(),
                          val venue: String = "")


data class SemanticScholarArticle(val ssid: String,
                                  val pmid: Int? = null,
                                  val citationList: List<String> = listOf(),
                                  val title: String = "",
                                  val abstract: String? = null,
                                  val year: Int? = null,
                                  val doi: String? = null,
                                  val keywords: String? = null,
                                  val source: PublicationSource? = null,
                                  val aux: ArticleAuxInfo = ArticleAuxInfo(),
                                  val crc32id: Int = 0)

