import org.jetbrains.exposed.sql.Table

object Publications : Table() {
    val pmid = integer("pmid").primaryKey()
    val year = integer("year").nullable()
    val title = varchar("title", 255)
    val abstract = text("abstract")
}

object Citations : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val pmid_citing = integer("pmid_citing")
    val pmid_cited = integer("pmid_cited")
}

object Keywords : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val keyword = varchar("keyword", 80)
}

object KeywordsPublications : Table() {
    val id = integer("id").primaryKey().autoIncrement()
    val pmid = integer("pmid")
    val keyword_id = integer("keyword_id")
    init {
        index(true, pmid, keyword_id)
    }
}