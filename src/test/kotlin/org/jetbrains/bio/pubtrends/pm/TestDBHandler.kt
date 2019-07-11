package org.jetbrains.bio.pubtrends.pm

import org.jetbrains.exposed.sql.selectAll
import org.jetbrains.exposed.sql.transactions.transaction

/**
 * This class is used to verify data which was stored in the database.
 */
class TestDBHandler(
    url: String,
    port: Int,
    database: String,
    user: String,
    password: String,
    resetDatabase: Boolean
) : PostgresqlDatabaseHandler(url, port, database, user, password, resetDatabase) {

    val articlesCount
        get() = transaction {
            PMPublications.selectAll().count()
        }

    val articlesPMIDList
        get() = transaction {
            PMPublications.selectAll().map { it[PMPublications.pmid] }
        }
}