import org.apache.logging.log4j.LogManager
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import java.io.*
import java.util.zip.GZIPInputStream
import javax.xml.parsers.SAXParserFactory

class PubmedCrawler {
    private val logger = LogManager.getLogger(PubmedCrawler::class)
    private val client = PubmedFTPClient()

    fun downloadBaseline(limit : Int) {
        var files = client.getBaselineXMLsList()
        if (files != null) {
            if (limit > 0) {
                files = files.subList(0, limit)
            }
            logger.info("Found ${files.size} baseline file(s)")
            for (file in files) {
                logger.info("${file.name}: Downloading... ")
                if (client.downloadBaselineFile(file.name, "data/")) {
                    logger.info("${file.name}: Unpacking... ")
                    if (unpack(file.name)) {
                        logger.info("${file.name}: SUCCESS")
                    } else {
                        logger.info("${file.name}: FAILURE")
                    }
                } else {
                    logger.info("${file.name}: FAILURE")
                }
            }
        }
    }

    fun downloadUpdateFile(name : String) {
        if (client.downloadUpdateFile(name, "data/")) {
            logger.info("$name: SUCCESS")
        } else {
            logger.info("$name: FAILURE")
        }
    }

//    fun parse(name : String) {
//        val spf = SAXParserFactory.newInstance()
//        spf.isNamespaceAware = true
//
//        val saxParser = spf.newSAXParser()
//        val pubmedXMLHandler = PubmedXMLHandler()
//        val xmlReader = saxParser.xmlReader
//        xmlReader.contentHandler = pubmedXMLHandler
//        try {
//            xmlReader.parse(InputSource(File(name).inputStream()))
//        } catch (e: SAXException) {
//            e.printStackTrace()
//        }
//
//        for (tag in pubmedXMLHandler.tags.keys) {
//            println("$tag ${pubmedXMLHandler.tags[tag]}")
//        }
//    }

    fun update() {
        // TODO: to config
        // Timestamp (ms) of 14th Oct 18 ~10:30 to download 1 new XML
        val lastCheck : Long = 1539513468000

        val files = client.getNewXMLsList(lastCheck)
        if (files != null) {
            logger.info("Found ${files.size} new file(s)")
            for (file in files) {
                logger.info("${file.name}: Downloading... ")
                if (client.downloadUpdateFile(file.name, "data/")) {
                    logger.info("${file.name}: Unpacking... ")
                    if (unpack(file.name)) {
                        logger.info("${file.name}: SUCCESS")
                    } else {
                        logger.info("${file.name}: FAILURE")
                    }
                } else {
                    logger.info("${file.name}: FAILURE")
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
                    logger.warn("Corrupted GZ archive. ")
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
}