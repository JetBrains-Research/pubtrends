package org.jetbrains.bio.pubtrends

import java.io.BufferedReader
import java.io.FileReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*

object Config {
    // Configure settings folder
    private val settingsRoot: Path = Paths.get(System.getProperty("user.home", ""), ".pubtrends")

    init {
        check(Files.exists(settingsRoot)) {
            "$settingsRoot should have been created by log4j"
        }
    }

    private val configPath: Path = settingsRoot.resolve("config.properties")

    init {
        check(Files.exists(configPath)) {
            "Config file not found, please modify and copy config.properties to $configPath"
        }
    }

    val config = Properties().apply {
        load(BufferedReader(FileReader(configPath.toFile())))
    }

    fun load() : Triple<Properties, Path, Path> {
        return Triple(config, configPath, settingsRoot)
    }
}