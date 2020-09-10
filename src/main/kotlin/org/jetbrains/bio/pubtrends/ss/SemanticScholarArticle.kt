package org.jetbrains.bio.pubtrends.ss

import org.apache.commons.codec.binary.Hex
import java.util.zip.CRC32

data class Author(val name: String = "")

data class Journal(
        val name: String = "",
        val volume: String = "",
        val pages: String = ""
)

data class Links(
        val s2Url: String = "",
        val s2PdfUrl: String = "",
        val pdfUrls: List<String> = listOf()
)

data class AuxInfo(
        val authors: List<Author> = listOf(),
        val journal: Journal = Journal(),
        val links: Links = Links(),
        val venue: String = ""
)

/**
 * Data class for [SSPublications]
 */
data class SemanticScholarArticle(
        val ssid: String,
        val pmid: Int? = null,
        val title: String = "",
        val abstract: String? = null,
        val year: Int? = null,
        val doi: String? = null,
        val keywords: String? = null,
        val aux: AuxInfo = AuxInfo(),
        val citations: List<String> = listOf()
)


private val crc32: CRC32 = CRC32()

fun crc32id(ssid: String): Int {
    crc32.reset()
    crc32.update(Hex.decodeHex(ssid.toCharArray()))
    return crc32.value.toInt()
}
