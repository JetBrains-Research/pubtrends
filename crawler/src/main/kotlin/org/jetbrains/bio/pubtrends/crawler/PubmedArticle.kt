package org.jetbrains.bio.pubtrends.crawler

data class PubmedArticle(var pmid : Int) {
    var year : Int? = null
    var title = ""
    var abstractText = ""
    val keywordList : MutableList<String> = mutableListOf()
    val citationList : MutableList<Int> = mutableListOf()

    fun description() : Map<String, String> {
        return mapOf("Year" to (year?.toString() ?: "undefined"),
                     "Title" to title,
                     "Abstract Text" to abstractText,
                     "Keywords" to keywordList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
                     "Citations" to citationList.joinToString(separator = ",", prefix = "\"", postfix = "\""))
    }
}