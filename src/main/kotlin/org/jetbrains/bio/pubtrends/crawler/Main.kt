package org.jetbrains.bio.pubtrends.crawler

fun main(args: Array<String>) {
//    TODO: CLI args processing
//    username and password for DB access

    val crawler = PubmedCrawler()
    crawler.update()
}