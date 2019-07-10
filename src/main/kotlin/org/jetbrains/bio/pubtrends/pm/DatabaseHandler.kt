package org.jetbrains.bio.pubtrends.pm

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.BatchInsertStatement
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs
import org.jetbrains.exposed.sql.transactions.TransactionManager
import org.jetbrains.exposed.sql.transactions.transaction

open class PostgresqlDatabaseHandler(
        url: String,
        port: Int,
        database: String,
        user: String,
        password: String,
        private val resetDatabase: Boolean

) : AbstractDBHandler {
    companion object Log4jSqlLogger : SqlLogger {
        private val logger = LogManager.getLogger(Log4jSqlLogger::class)

        override fun log(context: StatementContext, transaction: Transaction) {
            logger.debug("SQL: ${context.expandArgs(transaction)}")
        }
    }

    init {
        Database.connect(
                url = "jdbc:postgresql://$url:$port/$database",
                driver = "org.postgresql.Driver",
                user = user,
                password = password)

        transaction {
            addLogger(Log4jSqlLogger)

            if (resetDatabase) {
                SchemaUtils.drop(PMPublications, PMCitations)
            }

            exec("DROP TYPE IF EXISTS PublicationType; " +
                    "CREATE TYPE PublicationType " +
                    "AS ENUM ('ClinicalTrial', 'Dataset', 'TechnicalReport', 'Article', 'Review');")
            SchemaUtils.create(PMPublications, PMCitations)
        }
    }

    override fun store(articles: List<PubmedArticle>) {
        val citationsForArticle = articles.map { it.citationList.toSet().map { cit -> it.pmid to cit } }.flatten()

        transaction {
            addLogger(Log4jSqlLogger)

            PMPublications.batchInsertOnDuplicateKeyUpdate(articles,
                    listOf(PMPublications.year, PMPublications.title, PMPublications.abstract)) { batch, article ->
                batch[pmid] = article.pmid
                batch[year] = article.year
                batch[title] = article.title.take(PUBLICATION_MAX_TITLE_LENGTH)
                if (article.abstractText != "") {
                    batch[abstract] = article.abstractText
                }

                batch[keywords] = article.keywordList.joinToString(separator = ", ")
                batch[mesh] = article.meshHeadingList.joinToString(separator = ", ")

                batch[type] = article.type
                batch[doi] = article.doi
                batch[aux] = article.auxInfo
            }

            PMCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[PMCitations.pmidOut] = citation.first
                this[PMCitations.pmidIn] = citation.second
            }
        }
    }

    override fun delete(articlePMIDs: List<Int>) {
        transaction {
            addLogger(Log4jSqlLogger)

            PMPublications.deleteWhere { PMPublications.pmid inList articlePMIDs }
            PMCitations.deleteWhere {
                (PMCitations.pmidOut inList articlePMIDs) or (PMCitations.pmidIn inList articlePMIDs)
            }
        }
    }
}

class BatchInsertUpdateOnDuplicate(
        table: Table,
        private val onDupUpdate: List<Column<*>>
) : BatchInsertStatement(table, false) {
    override fun prepareSQL(transaction: Transaction): String {
        val onUpdateSQL = if (onDupUpdate.isNotEmpty()) {
            " ON CONFLICT (pmid) DO UPDATE SET ${
            onDupUpdate.joinToString {
                "${transaction.identity(it)} = Excluded.${transaction.identity(it)}"
            }}"
        } else ""
        return super.prepareSQL(transaction) + onUpdateSQL
    }
}

fun <T : Table, E> T.batchInsertOnDuplicateKeyUpdate(
        data: List<E>,
        onDupUpdateColumns: List<Column<*>>, body: T.(BatchInsertUpdateOnDuplicate, E) -> Unit
): List<Int> {
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
                    else -> error("can't find primary key of type Int or Long; " +
                            "map['$idCol']='$value' (where map='$it')")
                }
            }
        }
    }.orEmpty()
}