data class PubmedArticle(var pmid : Int) {
    var year : Int? = null
    var title = ""
    var abstractText = ""
    val keywordList : MutableList<String> = mutableListOf()
    val citationList : MutableList<Int> = mutableListOf()

    fun toList() : List<String> {
        return listOf(pmid.toString(), year?.toString() ?: "", "\"$title\"", "\"$abstractText\"",
                      keywordList.joinToString(separator = ",", prefix = "\"", postfix = "\""),
                      citationList.joinToString(separator = ",", prefix = "\"", postfix = "\""))
    }
}