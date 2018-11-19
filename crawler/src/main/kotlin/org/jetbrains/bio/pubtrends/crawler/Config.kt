package org.jetbrains.bio.pubtrends.crawler

import java.util.*

object Config {
    private val prop = Properties()

    init {
        val propertiesStream = PubmedCrawler::class.java.classLoader.getResourceAsStream("config.properties")
        propertiesStream.use {
            prop.load(it)
        }
    }

    operator fun get(key: String) : String {
        return prop[key].toString()
    }
}