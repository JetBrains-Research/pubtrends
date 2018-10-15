import org.xml.sax.InputSource
import java.io.*
import java.util.zip.GZIPInputStream
import javax.xml.parsers.SAXParserFactory

fun downloadBaseline(limit : Int) {
    val client = PubmedFTPClient()
    var files = client.getBaselineXMLsList()
    if (files != null) {
        if (limit > 0) {
            files = files.subList(0, limit)
        }
        println("Found ${files.size} baseline file(s)")
        for (file in files) {
            print("${file.name}: Downloading... ")
            if (client.downloadBaselineFile(file.name, "data/")) {
                print("Unpacking... ")
                if (unpack(file.name)) {
                    println("SUCCESS")
                } else {
                    println("FAILURE")
                }
            } else {
                println("${file.name}: FAILURE")
            }
        }
    }
}

fun downloadUpdateFile(name : String) {
    val client = PubmedFTPClient()
    if (client.downloadUpdateFile(name, "data/")) {
        println("$name: SUCCESS")
    } else {
        println("$name: FAILURE")
    }
}
/*
fun parse(name : String) {
    val spf = SAXParserFactory.newInstance()
    spf.isNamespaceAware = true

    val saxParser = spf.newSAXParser()
    val pubmedXMLHandler = PubmedXMLHandler()
    val xmlReader = saxParser.xmlReader
    xmlReader.contentHandler = pubmedXMLHandler
    xmlReader.parse(InputSource(File(name).inputStream()))

    for (tag in pubmedXMLHandler.tags.keys) {
        println("$tag - ${pubmedXMLHandler.tags[tag]}")
    }
}
*/
fun update() {
    // TODO: to config
    // Timestamp (ms) of 14th Oct 18 ~10:30 to download 1 new XML
    val lastCheck : Long = 1539513468000

    val client = PubmedFTPClient()
    val files = client.getNewXMLsList(lastCheck)
    if (files != null) {
        println("Found ${files.size} new file(s)")
        for (file in files) {
            print("${file.name}: Downloading... ")
            if (client.downloadUpdateFile(file.name, "data/")) {
                print("Unpacking... ")
                if (unpack(file.name)) {
                    println("SUCCESS")
                } else {
                    println("FAILURE")
                }
            } else {
                println("${file.name}: FAILURE")
            }
        }
    }
}

fun unpack(archiveName : String) : Boolean {
    val archive = File("data/$archiveName")
    val originalName = "data/${archiveName.substringBefore(".gz")}"
    val bufferSize = 1024
    var safeUnpack = true

    GZIPInputStream(BufferedInputStream(FileInputStream("data/$archiveName"))).use { inputStream ->
        BufferedOutputStream(FileOutputStream(originalName)).use { outputStream ->
            try {
                inputStream.copyTo(outputStream, bufferSize)
            } catch (e : EOFException) {
                print("Warning: Corrupted GZ archive. ")
                e.printStackTrace()
                safeUnpack = false
            }
        }
    }

    if (safeUnpack) {
        archive.delete()
    }

    return safeUnpack
}

fun main(args: Array<String>) {
//    downloadBaseline(10)
//    parse("data/0001.xml")
    update()
}