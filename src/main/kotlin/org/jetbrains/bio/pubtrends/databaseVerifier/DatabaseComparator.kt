package org.jetbrains.bio.pubtrends.databaseVerifier

import org.jetbrains.exposed.sql.Table


class DatabaseComparator {

    fun compareTables (first: Table, second: Table) {
        // cant find sql "except" operator at "exposed"

        // in progress..
    }
}
