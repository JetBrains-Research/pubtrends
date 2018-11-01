import org.jetbrains.exposed.sql.Table

object Publications : Table() {
    val pmid = integer("pmid").primaryKey()
    val year = integer("year").nullable()
    val title = varchar("title", 255)
    val abstract = text("abstract")
}

data class Publication(
        val pmid : Int,
        val year : Int?,
        val title : String,
        val abstract : String
)

object Citations : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val pmid_citing = integer("pmid_citing")
    val pmid_cited = integer("pmid_cited")
}

data class Citation(
        val id : Int,
        val pmid_citing : Int,
        val pmid_cited : Int
)

object Keywords : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val keyword = varchar("keyword", 80)
}

data class Keyword(
        val id : Int,
        val keyword: String
)

object KeywordsPublications : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val pmid = integer("pmid")
    val keyword_id = integer("keyword_id")
    init {
        index(true, pmid, keyword_id)
    }
}