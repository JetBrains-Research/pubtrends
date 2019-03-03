package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.BatchInsertStatement
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.TransactionManager
import org.jetbrains.exposed.sql.transactions.transaction

class DatabaseHandler : AbstractDBHandler {
    companion object Log4jSqlLogger : SqlLogger {
        private val logger = LogManager.getLogger(Log4jSqlLogger::class)

        override fun log(context: StatementContext, transaction: Transaction) {
            logger.debug("SQL: ${context.expandArgs(transaction)}")
        }
    }

    init {
        Database.connect("jdbc:postgresql://${Config["url"]}:${Config["port"]}/${Config["database"]}",
                         driver = "org.postgresql.Driver",
                user = Config["username"],
                password = Config["password"])

        transaction {
            addLogger(Log4jSqlLogger)

            if (Config["resetDatabase"].toBoolean()) {
                SchemaUtils.drop(Publications, Citations, Keywords, KeywordsPublications)
            }

            SchemaUtils.create(Publications, Citations, Keywords, KeywordsPublications)
        }
    }

    override fun store(articles: List<PubmedArticle>) {
        // val keywordsForArticle = articles.map { it.keywordList.map { kw -> it.pmid to kw } }.flatten()
        // val keywordSet = keywordsForArticle.mapTo(hashSetOf()) { it.second }
        logger.info("Storing ${articles.size} articles...")
        val citationsForArticle = articles.map { it.citationList.toSet().map { cit -> it.pmid to cit } }.flatten()

        transaction {
            addLogger(Log4jSqlLogger)

            Publications.batchInsertOnDuplicateKeyUpdate(articles,
                    listOf(Publications.year, Publications.title, Publications.abstract)) { batch, article ->
                batch[Publications.pmid] = article.pmid
                batch[Publications.year] = article.year
                batch[Publications.title] = article.title.take(1023)
                if (article.abstractText != "") {
                    batch[Publications.abstract] = article.abstractText
                }
            }

            Citations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[Citations.pmidCiting] = citation.first
                this[Citations.pmidCited] = citation.second
            }

            /*val keywordIds = Keywords.batchInsert(keywordSet, ignore = true) { keyword ->
                this[Keywords.keyword] = keyword
            }

            val keywordIdsMap = keywordSet.zip(keywordIds) { kw, rs ->
                kw to try {
                    rs.getValue(Keywords.id) as Int
                } catch (e : NoSuchElementException) {
                    Keywords.select { Keywords.keyword eq kw }.map { it[Keywords.id] }[0]
                }
            }.toMap()
            keywordIdsMap.toSortedMap().forEach {
                println("${it.key} - ${it.value}")
            }
            val keywordIdsForArticle = keywordsForArticle.map { it.first to (keywordIdsMap[it.second] ?: 0) }

            KeywordsPublications.batchInsert(keywordIdsForArticle) { keywordId ->
                this[KeywordsPublications.pmid] = keywordId.first
                this[KeywordsPublications.keywordId] = keywordId.second
            }*/
        }
    }
}

class BatchInsertUpdateOnDuplicate(table: Table, private val onDupUpdate: List<Column<*>>): BatchInsertStatement(table, false) {
    override fun prepareSQL(transaction: Transaction): String {
        val onUpdateSQL = if(onDupUpdate.isNotEmpty()) {
            " ON CONFLICT (pmid) DO UPDATE SET " + onDupUpdate.joinToString { "${transaction.identity(it)} = Excluded.${transaction.identity(it)}" }
        } else ""
        return super.prepareSQL(transaction) + onUpdateSQL
    }
}

fun <T: Table, E> T.batchInsertOnDuplicateKeyUpdate(data: List<E>, onDupUpdateColumns: List<Column<*>>, body: T.(BatchInsertUpdateOnDuplicate, E) -> Unit): List<Int> {
    return data.takeIf { it.isNotEmpty() }?.let {
        val insert = BatchInsertUpdateOnDuplicate(this, onDupUpdateColumns)
        data.forEach {
            insert.addBatch()
            body(insert, it)
        }
        TransactionManager.current().exec(insert)
        columns.firstOrNull { it.columnType.isAutoInc }?.let { idCol ->
            insert.generatedKey?.mapNotNull {
                val value = it[idCol]
                when (value) {
                    is Long -> value.toInt()
                    is Int -> value
                    null -> null
                    else -> error("can't find primary key of type Int or Long; map['$idCol']='$value' (where map='$it')")
                }
            }
        }
    }.orEmpty()
}