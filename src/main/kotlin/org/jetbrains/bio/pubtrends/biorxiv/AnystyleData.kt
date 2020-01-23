package org.jetbrains.bio.pubtrends.biorxiv

import com.google.gson.annotations.SerializedName
import org.jetbrains.bio.pubtrends.pm.PubmedArticle

data class AnystyleData(
        val references: List<AnystyleReference> = listOf()
)

data class AnystyleAuthor(
        @SerializedName("family")
        val familyName: String = "",
        @SerializedName("given")
        val givenName: String = "",
        val particle: String = "",
        @SerializedName("others")
        val otherAuthors: String = ""
) {
    override fun toString() : String {
        // Corresponds to "et al."
        if (otherAuthors.isNotEmpty()) {
            return ""
        }
        return "$givenName $particle $familyName"
    }
}

/**
 * This class represents JSON structure of `anystyle` output for a separate reference, which
 * contains at least the following fields: authors, title, volume, date, pages, type,
 * container-title (journal/conference), issue, volume, doi, url
 *
 * No full specification of `anystyle` output format was found.
 *
 * However, it is meaningful to parse only the fields that are present in PubmedArticle.
 */
data class AnystyleReference(
        @SerializedName("author")
        val authors: List<AnystyleAuthor> = listOf(),
        @SerializedName("container-title")
        val journal: List<String> = listOf(),
        val title: List<String> = listOf(),
        val date: List<String> = listOf(),
        val doi: List<String> = listOf()
) {
    fun toPubmedArticle() : PubmedArticle {
        title.forEach {
            it.replace("”", "")
        }
        return PubmedArticle(pmid = 0, title = title.joinToString(""))
    }
}