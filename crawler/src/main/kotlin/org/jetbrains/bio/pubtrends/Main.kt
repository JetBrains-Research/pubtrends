package org.jetbrains.bio.pubtrends

import org.jetbrains.bio.pubtrends.crawler.PubmedCrawler

fun main(args: Array<String>) {
    val crawler = PubmedCrawler()
    crawler.update()
}