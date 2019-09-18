package org.jetbrains.bio.pubtrends

import org.jetbrains.bio.pubtrends.pm.Neo4jDatabaseHandler
import java.io.BufferedReader
import java.io.FileReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import kotlin.system.exitProcess

/**
 * @author Oleg Shpynov
 * @date 2019-07-22
 */
class Main {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            println("Welcome to Pubtrends!")

            // Configure settings folder
            val settingsRoot: Path = Paths.get(System.getProperty("user.home", ""), ".pubtrends")
            check(Files.exists(settingsRoot)) {
                "$settingsRoot should have been created by log4j"
            }
            println("Settings folder $settingsRoot")

            val configPath: Path = settingsRoot.resolve("config.properties")
            if (Files.notExists(configPath)) {
                println("Config file not found, please modify and copy config.properties to $configPath")
                exitProcess(1)
            }

            val config = Properties().apply {
                load(BufferedReader(FileReader(configPath.toFile())))
            }

            val dbHandler = Neo4jDatabaseHandler(config["neo4jurl"].toString(), 7687,
                    config["neo4jusername"].toString(), config["neo4jpassword"].toString(), false)
            dbHandler.init()
            dbHandler.close()
        }
    }
}