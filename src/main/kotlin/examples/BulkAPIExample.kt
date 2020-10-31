package examples

import org.jetbrains.bio.pubtrends.arxiv.ArxivXMLDomParser

import com.github.kittinunf.fuel.httpGet
import com.github.kittinunf.result.Result
import java.io.File

fun main() {
    val requestURL = "http://export.arxiv.org/oai2?verb=ListRecords&from=2018-03-20&metadataPrefix=arXiv"
    val outputXMLFile = File("files/response.xml")
    val outputFile = File("files/metadata.txt")
    outputFile.writeText("")
    val (_, _, result) = requestURL
        .httpGet()
        .timeoutRead(60000)
        .responseString()
    when (result) {
        is Result.Failure -> {
            val ex = result.getException()
            println(ex)
            return
        }
        is Result.Success -> {
            println("Success")
            val data = result.get()
            outputXMLFile.writeText(data)
            for (elem in ArxivXMLDomParser.parseArxivRecords(data).first) {
                outputFile.appendText(elem.toString())
            }
        }
    }
}