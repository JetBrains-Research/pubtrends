package org.jetbrains.bio.pubtrends.pm

data class Author (var name : String = "",
                   val affiliation : MutableList<String> = mutableListOf()) { }

data class Journal (var name : String = "") { }

data class DatabankEntry (var name: String = "",
                          val accessionNumber : MutableList<String> = mutableListOf()) { }

data class ArticleAuxInfo (val authors : MutableList<Author> = mutableListOf(),
                           val databanks : MutableList<DatabankEntry> = mutableListOf(),
                           val journal : Journal = Journal(),
                           var language : String = "") { }

data class PubmedArticle(var pmid : Int = 0,
                         var year : Int? = null,
                         var title : String = "",
                         var abstractText : String = "",
                         val keywordList : MutableList<String> = mutableListOf(),
                         val citationList : MutableList<Int> = mutableListOf(),
                         val databankEntryList : MutableList<DatabankEntry> = mutableListOf(),
                         val meshHeadingList : MutableList<String> = mutableListOf(),
                         var type : PublicationType = PublicationType.Article,
                         var doi : String = "",
                         val auxInfo : ArticleAuxInfo = ArticleAuxInfo()) {

    fun description() : Map<String, String> {
        return mapOf("PMID" to pmid.toString(),
                "Year" to (year?.toString() ?: "undefined"),
                "Title" to title,
                "Type" to type.name,
                "Abstract Text" to abstractText,
                "DOI" to doi,
                "Keywords" to keywordList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
                "Citations" to citationList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
                "Other information" to auxInfo.toString())
    }
}