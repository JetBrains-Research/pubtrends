package org.jetbrains.bio.pubtrends.ss

import org.apache.commons.codec.binary.Hex
import java.util.zip.CRC32

data class Author(var name: String = "")

data class Journal(var name: String = "", var volume: String = "", var pages: String = "")

data class Links(var s2Url: String = "", var s2PdfUrl: String = "", var pdfUrls: List<String> = listOf())

data class ArticleAuxInfo(val authors: List<Author> = listOf(),
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
                                  val aux: ArticleAuxInfo = ArticleAuxInfo())


private val crc32: CRC32 = CRC32()

fun crc32id(ssid: String): Int {
    crc32.reset()
    crc32.update(Hex.decodeHex(ssid.toCharArray()))
    return crc32.value.toInt()
}
