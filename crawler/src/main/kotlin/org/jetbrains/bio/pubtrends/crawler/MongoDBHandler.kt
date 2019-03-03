package org.jetbrains.bio.pubtrends.crawler

import org.litote.kmongo.*

class MongoDBHandler : AbstractDBHandler {
    private val client = KMongo.createClient(Config["mongoUrl"], Config["mongoPort"].toInt())
    private val database = client.getDatabase(Config["mongoDatabase"])
    private val collection = database.getCollection<PubmedArticle>("publications")

    override fun store(articles: List<PubmedArticle>) {
        collection.insertMany(articles)
    }
}