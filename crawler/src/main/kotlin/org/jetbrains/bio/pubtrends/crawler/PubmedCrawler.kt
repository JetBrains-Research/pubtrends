package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import java.io.*
import java.sql.Timestamp
import java.util.zip.GZIPInputStream

class PubmedCrawler {
    private val logger = LogManager.getLogger(PubmedCrawler::class)
    private val ftpHandler = PubmedFTPHandler()
    private val xmlParser = PubmedXMLParser(DatabaseHandler())

    private val lastCheck = Config["lastModification"].toLong()
    private val lastId = Config["lastId"].toInt()
    internal val tempDirectory = createTempDir()
    //    private val lastCheck : Long = 1539513468000

    init {
        if (lastCheck.compareTo(0) != 0) {
            logger.info("Last modification: ${Timestamp(lastCheck).toLocalDateTime()}")
        }

        if (lastId > 0) {
            logger.info("Last downloaded file: pubmed19n${lastId.toString().padStart(4, '0')}.xml.gz")
        }

        if (tempDirectory.exists()) {
            logger.info("Created temporary directory: ${tempDirectory.absolutePath}")
        }
    }

    fun update() {
        try {
            val (baselineFiles, updateFiles) = ftpHandler.fetch(lastCheck, lastId)
            logger.info("Found ${baselineFiles.size + updateFiles.size} new file(s)")

            downloadFiles(baselineFiles, isBaseline = true)
            downloadFiles(updateFiles, isBaseline = false)
        } catch (e : IOException) {
            logger.fatal("Failed to connect to the server. Error message: ${e.printStackTrace()}")
        } finally {
            if (tempDirectory.exists()) {
                logger.info("Deleting directory: ${tempDirectory.absolutePath}")
                tempDirectory.deleteRecursively()
            }
            if (Config["gatherStats"].toBoolean()) {
                File("tag_stats.csv").outputStream().bufferedWriter().use {
                    xmlParser.pubmedXMLHandler.tags.iterator().forEach { tag ->
                        it.write("${tag.key} ${tag.value}\n")
                    }
                }
            }
        }
    }

    private fun unpack(archiveName : String) : Boolean {
        val archive = File(archiveName)
        val originalName = archiveName.substringBefore(".gz")
        val bufferSize = 1024
        var safeUnpack = true

        logger.info("$archiveName: Unpacking...")

        GZIPInputStream(BufferedInputStream(FileInputStream(archiveName))).use {
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
            val localArchiveName = "${tempDirectory.absolutePath}/$it"
            val localName = localArchiveName.substringBefore(".gz")

            logger.info("$localArchiveName: Downloading...")

            val downloadSuccess = if (isBaseline) ftpHandler.downloadBaselineFile(it, tempDirectory.absolutePath)
                                  else ftpHandler.downloadUpdateFile(it, tempDirectory.absolutePath)
            var overallSuccess = false

            if (downloadSuccess && unpack(localArchiveName)) {
                overallSuccess = xmlParser.parse(localName)
                File(localName).delete()
            }

            logger.info("$localName: ${if (overallSuccess) "SUCCESS" else "FAILURE"}")
        }
    }
}