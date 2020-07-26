import com.preprint.server.core.arxiv.ArxivAPI

val recordIds = listOf("1507.11111", "1604.08289", "1608.08082", "1403.5117")

fun main() {
    for (elem in ArxivAPI.getArxivRecords(recordIds)) {
        println(elem)
    }
}