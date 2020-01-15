package org.jetbrains.bio.pubtrends

/**
 * @author Oleg Shpynov
 * @date 2019-07-22
 */
class Main {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            println("""
Welcome to Pubtrends!
Please use PubmedLoader or SemanticScholarLoader to fill database.
See README.md for deployment instructions.
""")
        }
    }
}