package examples

import com.preprint.server.core.crossref.CrossRefApi

fun main() {
    val ref = "Napiwotzki, R., Koester, D., & Nelemans, G., et al., 2002, A&A, 386, 957"
    val recordList = CrossRefApi.findRecord(ref)
    println(recordList)
}