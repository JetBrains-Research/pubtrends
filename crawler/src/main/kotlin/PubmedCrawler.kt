import org.apache.logging.log4j.LogManager
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import java.io.*
import java.sql.Timestamp
import java.util.zip.GZIPInputStream
import javax.xml.parsers.SAXParserFactory

class PubmedCrawler {
    private val logger = LogManager.getLogger(PubmedCrawler::class)
    private val client = PubmedFTPHandler()
    private val dbHandler = DatabaseHandler("biolabs", "pubtrends", reset = true)
    private val spf = SAXParserFactory.newInstance()

    init {
        spf.isNamespaceAware = true
    }

    private val saxParser = spf.newSAXParser()
    val pubmedXMLHandler = PubmedXMLHandler()
    private val xmlReader = saxParser.xmlReader

    init {
        xmlReader.contentHandler = pubmedXMLHandler
    }

    private val lastCheck = dbHandler.lastModification
//    private val lastCheck : Long = 1539513468000

    init {
        if (lastCheck.compareTo(0) != 0) {
            logger.info("Last modification: ${Timestamp(lastCheck).toLocalDateTime()}")
        }
    }

    fun update() {
        val files = client.fetch(lastCheck)

        val baselineFiles = files.first
        val updateFiles = files.second
        logger.info("Found ${baselineFiles.size + updateFiles.size} new file(s)")
        downloadFiles(baselineFiles, isBaseline = true)
        downloadFiles(updateFiles, isBaseline = false)
    }

    private fun unpack(archiveName : String) : Boolean {
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

    private fun downloadFiles(files : List<String>, isBaseline : Boolean) {
        files.forEach {
            logger.info("$it: Downloading...")

            val downloadSuccess = when (isBaseline) {
                true -> client.downloadBaselineFile(it, "data/")
                false -> client.downloadUpdateFile(it, "data/")
            }

            var overallSuccess = true
            if (downloadSuccess) {
                logger.info("$it: Unpacking...")
                overallSuccess = overallSuccess && unpack(it)
            }

            val name = it.substringBefore(".gz")
            if (overallSuccess) {
                logger.info("$it: Parsing...")
                overallSuccess = overallSuccess && parse(name)
            }

            if (overallSuccess) {
                logger.info("$it: Storing...")
                pubmedXMLHandler.articles.forEach {article ->
                    dbHandler.store(article)
                }
            }

            File(name).delete()
            logger.info("$it: ${if (overallSuccess) "SUCCESS" else "FAILURE"}")
        }
    }

    fun parse(name : String) : Boolean {
        try {
            val localName = "../data/$name"
            logger.debug("File location: ${File(localName).absolutePath}")
            xmlReader.parse(InputSource(File(localName).inputStream()))
        } catch (e: SAXException) {
            e.printStackTrace()
        }

        return true
    }
}