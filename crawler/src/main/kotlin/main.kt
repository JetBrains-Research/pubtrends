import org.xml.sax.EntityResolver
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import java.io.*
import java.util.zip.GZIPInputStream
import javax.xml.parsers.SAXParserFactory

fun main(args: Array<String>) {
//    TODO: CLI args processing

    val crawler = PubmedCrawler()
//    crawler.downloadBaseline(10)
//    crawler.parse("data/0001.xml")
    crawler.update()
}