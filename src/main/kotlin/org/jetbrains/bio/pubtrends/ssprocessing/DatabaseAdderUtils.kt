package org.jetbrains.bio.pubtrends.ssprocessing

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.transactions.transaction
import java.lang.IndexOutOfBoundsException

object DatabaseAdderUtils {
    private val logger = LogManager.getLogger(DatabaseAdderUtils::class)

    fun addArticles(articles: MutableList<SemanticScholarArticle>) {
        val keywordSet = articles.map{it.keywordList}.flatten().toSet()

        transaction {
//            addLogger(StdOutSqlLogger)

            val articlesId = SSPublications.batchInsert(articles) { article ->
                this[SSPublications.ssid] = article.ssid
                this[SSPublications.pmid] = article.pmid
                this[SSPublications.abstract] = article.abstract
                this[SSPublications.title] = article.title
                this[SSPublications.year] = article.year
                this[SSPublications.doi] = article.doi
                this[SSPublications.sourceEnum] = article.source
                this[SSPublications.aux] = article.aux
            }


            val articlesIdMatch =  articles.map{it.ssid}
                    .zip(articlesId.map{ it[SSPublications.id].toString().toInt()}).toMap()  // is correct only if publications doesn't repeat


            val keywordsForArticle = articles.map { it.keywordList.toSet().map {
                keywords -> articlesIdMatch[it.ssid] to keywords }
            }.flatten()

            //keywordsArticle example: [(23001, Metabolic Biotransformation), (23001, Renal Elimination), (23001, Sulfamethizole)]

            SSKeywords.batchInsert(keywordSet, ignore = true) { keyword ->
                this[SSKeywords.keyword] = keyword
            }

            // result of batch insert is incorrect in case of insert already existing value

            val keywordIdMap = keywordSet.map{ kw ->
                kw to SSKeywords.select { SSKeywords.keyword eq kw }.map { it[SSKeywords.id] }[0]
            }.toMap()

            //keywordIdMap example: {Big data=752, Choose (action)=505, Clustering high-dimensional data=109904}

            SSKeywordsPublications.batchInsert(keywordsForArticle) { (p_id, keyword) ->
                this[SSKeywordsPublications.sspid] = p_id!!
                this[SSKeywordsPublications.sskid] = keywordIdMap.getValue(keyword)
            }
        }
    }

    fun addCitations(articles: MutableList<SemanticScholarArticle>) {
        val articlesIdOut = articles.map{it.ssid}
        val articlesIdIn = articles.map {it.citationList}.flatten()
        val articlesSet = articlesIdOut.union(articlesIdIn)

        transaction {
//            addLogger(StdOutSqlLogger)

            val articleIdMap = articlesSet.map{ ssid ->
                ssid to SSPublications.select { SSPublications.ssid eq ssid }.map { it[SSPublications.id] }[0]
            }.toMap()


            val citationsForArticle = articles
                    .map { it.citationList.map { cit -> articleIdMap[it.ssid] to articleIdMap[cit] } }
                    .flatten()


            SSCitations.batchInsert(citationsForArticle, ignore = true) { citation ->
                this[SSCitations.id_out] = citation.first!!
                this[SSCitations.id_in] = citation.second!!
            }

            }

    }

/*    fun addPmidCitations() {
        transaction {
            addLogger(StdOutSqlLogger)

            val idMatch1 = IdMatch.alias("im1")
            val idMatch2 = IdMatch.alias("im2")

            PmidCitationsFromSS.insertIgnore(SemanticScholarCitations
                    .innerJoin(idMatch1, {SSCitations.idCiting}, {idMatch1[IdMatch.ssid]})
                    .innerJoin(idMatch2, {SSCitations.idCited}, {idMatch2[IdMatch.ssid]})
                    .slice(listOf(idMatch1[IdMatch.pmid], idMatch2[IdMatch.pmid]))
                    .selectAll())


        }
    }
*/

}