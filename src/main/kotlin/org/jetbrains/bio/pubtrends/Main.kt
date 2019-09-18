package org.jetbrains.bio.pubtrends

import org.jetbrains.bio.pubtrends.pm.Neo4jDatabaseHandler

/**
 * @author Oleg Shpynov
 * @date 2019-07-22
 */
class Main {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            println("Welcome to Pubtrends!")
            val dbHandler = Neo4jDatabaseHandler("localhost", 7687, "neo4j",
                    "deve1oper", false)
            dbHandler.init()
            dbHandler.close()
        }
    }
}