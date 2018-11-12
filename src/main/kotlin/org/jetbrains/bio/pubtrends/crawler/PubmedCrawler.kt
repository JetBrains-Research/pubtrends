package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import java.io.*
import java.sql.Timestamp
import java.util.zip.GZIPInputStream
import javax.xml.parsers.SAXParserFactory

class PubmedCrawler {
    private val logger = LogManager.getLogger(PubmedCrawler::class)
    private val ftpHandler = PubmedFTPHandler()
    private val dbHandler = DatabaseHandler("biolabs", "pubtrends", reset = false)
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
    private val tempDirectory = File("tmp")
//    private val lastCheck : Long = 1539513468000

    init {
        if (lastCheck.compareTo(0) != 0) {
            logger.info("Last modification: ${Timestamp(lastCheck).toLocalDateTime()}")
        }

        if (!tempDirectory.exists()) {
            tempDirectory.mkdir()
            logger.info("Created directory for file download: ${tempDirectory.absolutePath}")
        }
    }

    fun update() {
        val files = ftpHandler.fetch(lastCheck)

        val baselineFiles = files.first
        val updateFiles = files.second
        logger.info("Found ${baselineFiles.size + updateFiles.size} new file(s)")

        downloadFiles(baselineFiles, isBaseline = true)
        downloadFiles(updateFiles, isBaseline = false)

        if (!tempDirectory.exists()) {
            if (tempDirectory.list().isNotEmpty()) {
                logger.warn("Temporary directory is not empty.")
            } else {
                logger.info("Deleting directory: ${tempDirectory.absolutePath}")
                tempDirectory.deleteOnExit()
            }
        }
    }

    private fun unpack(archiveName : String) : Boolean {
        val archive = File("${tempDirectory.name}/$archiveName")
        val originalName = "${tempDirectory.name}/${archiveName.substringBefore(".gz")}"
        val bufferSize = 1024
        var safeUnpack = true

        logger.info("$archiveName: Unpacking...")

        GZIPInputStream(BufferedInputStream(FileInputStream("${tempDirectory.name}/$archiveName"))).use {
            inputStream -> BufferedOutputStream(FileOutputStream(originalName)).use { outputStream ->
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

            val downloadSuccess = if (isBaseline) ftpHandler.downloadBaselineFile(it, tempDirectory.name)
                                  else ftpHandler.downloadUpdateFile(it, tempDirectory.name)
            var overallSuccess = false
            val name = it.substringBefore(".gz")

            if (downloadSuccess && unpack(it)) {
                if (parse(name)) {
                    logger.info("$it: Storing...")
                    pubmedXMLHandler.articles.forEach {article ->
                        dbHandler.store(article)
                    }
                    overallSuccess = true
                }

                File(name).delete()
            }

            logger.info("$name: ${if (overallSuccess) "SUCCESS" else "FAILURE"}")
        }
    }

    fun parse(name : String) : Boolean {
        logger.info("$name: Parsing...")

        try {
            val localName = "${tempDirectory.name}/$name"
            logger.debug("File location: ${File(localName).absolutePath}")
            xmlReader.parse(InputSource(File(localName).inputStream()))
        } catch (e: SAXException) {
            e.printStackTrace()
        }

        return true
    }
}