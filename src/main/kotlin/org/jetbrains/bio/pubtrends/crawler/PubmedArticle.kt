package org.jetbrains.bio.pubtrends.crawler

enum class ArticleTypes {
    ClinicalTrial,
    Dataset,
    TechnicalReport,
    Review,
    Article
}

data class Author (var name : String = "",
                   val affiliation : MutableList<String> = mutableListOf()) { }

data class Journal (var name : String = "") { }

data class ArticleAuxInfo (val authors : MutableList<Author> = mutableListOf(),
                           val journal : Journal = Journal(),
                           var language : String = "") { }

data class DatabankEntry (val name: String = "", val accessionNumber : String = "") { }

data class PubmedArticle(var pmid : Int = 0,
                         var year : Int? = null,
                         var title : String = "",
                         var abstractText : String = "",
                         val keywordList : MutableList<String> = mutableListOf(),
                         val citationList : MutableList<Int> = mutableListOf(),
                         val databankEntryList : MutableList<DatabankEntry> = mutableListOf(),
                         val meshHeadingList : MutableList<String> = mutableListOf(),
                         var type : ArticleTypes = ArticleTypes.Article,
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