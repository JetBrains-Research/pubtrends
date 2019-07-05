package org.jetbrains.bio.pubtrends.crawler

import kotlin.reflect.jvm.internal.impl.load.kotlin.JvmType

data class Author (var name : String = "",
                   val affiliation : MutableList<String> = mutableListOf()) { }

data class Journal (var name : String = "") { }

data class ArticleAuxInfo (val authors : MutableList<Author> = mutableListOf(),
                           val journal : Journal = Journal(),
                           var language : String = "") { }

data class PubmedArticle(var pmid : Int) {
    var year : Int? = null
    var title = ""
    var abstractText = ""
    val keywordList : MutableList<String> = mutableListOf()
    val citationList : MutableList<Int> = mutableListOf()
    var type = ""
    var doi = ""
    val auxInfo = ArticleAuxInfo()

    fun description() : Map<String, String> {
        return mapOf("Year" to (year?.toString() ?: "undefined"),
                     "Title" to title,
                     "Abstract Text" to abstractText,
                     "Keywords" to keywordList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
                     "Citations" to citationList.joinToString(separator = ",", prefix = "\"", postfix = "\""))
    }
}