import org.jetbrains.exposed.dao.EntityID
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.transactions.transaction

class DatabaseHandler(username : String, password : String) {
    init {
        Database.connect("jdbc:postgresql://localhost:5432/pubmed",
                         driver = "org.postgresql.Driver",
                         user = username,
                         password = password)

        transaction {
            addLogger(Log4jSqlLogger)

            SchemaUtils.create(Publications)
            SchemaUtils.create(Citations)
            SchemaUtils.create(Keywords)
            SchemaUtils.create(KeywordsPublications)
        }
    }

    fun store(article : PubmedArticle) {
        transaction {
            addLogger(Log4jSqlLogger)

            Publications.insert {
                it[pmid] = article.pmid
                it[year] = article.year
                it[title] = article.title
                it[abstract] = article.abstractText
            }

            val keywordIds = Keywords.batchInsert(article.keywordList, ignore = true) {keyword ->
                this[Keywords.keyword] = keyword
            }

            KeywordsPublications.batchInsert(keywordIds) {keywordId ->
                this[KeywordsPublications.pmid] = article.pmid
                this[KeywordsPublications.keyword_id] = keywordId.getValue(Keywords.id) as Int
            }

            Citations.batchInsert(article.citationList) { citation ->
                this[Citations.pmid_citing] = article.pmid
                this[Citations.pmid_cited] = citation
            }
        }
    }
}