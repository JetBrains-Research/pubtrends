package org.jetbrains.bio.pubtrends.crawler

import org.apache.logging.log4j.LogManager
import java.io.*
import java.nio.file.Files
import java.nio.file.Path
import java.sql.Timestamp
import java.time.Instant
import java.util.*
import java.util.zip.GZIPInputStream

class PubmedCrawler(
        dbHandler: PubmedXMLHandler,
        private val collectStats: Boolean,
        private val statsPath: Path,
        private val progressPath: Path
) {

    private val logger = LogManager.getLogger(PubmedCrawler::class)

    init {
        if (collectStats) {
            logger.info("Collecting stats in $statsPath")
        }
    }

    private val ftpHandler = PubmedFTPHandler()
    private val xmlParser = PubmedXMLParser(dbHandler)
    private lateinit var tempDirectory: File

    /**
     * @return false if update not required
     */
    fun update(lastCheckCmd: Long?, lastIdCmd: Int?): Boolean {
        val (lastCheckP, lastIdP) = loadLastProgress()
        val lastCheck = lastCheckCmd ?: lastCheckP ?: 0L
        val lastId = lastIdCmd ?: lastIdP ?: 0

        if (lastCheck != 0L) {
            logger.info("Last check: ${Timestamp(lastCheck).toLocalDateTime()}")
        }

        if (lastId > 0) {
            logger.info("Last downloaded file: ${PubmedFTPHandler.idToPubmedFile(lastId)}")
        }

        tempDirectory = createTempDir()
        logger.info("Created temporary directory: ${tempDirectory.absolutePath}")
        try {
            val (baselineFiles, updateFiles) = ftpHandler.fetch(lastCheck, lastId)
            val baselineSize = baselineFiles.size
            val updatesSize = updateFiles.size
            logger.info("Found ${baselineSize + updatesSize} new file(s)\n" +
                    "Baseline: $baselineSize, Updates: $updatesSize")
            if (baselineSize + updatesSize == 0) {
                return false
            }
            logger.info("Processing baseline")
            downloadFiles(baselineFiles, isBaseline = true)
            logger.info("Processing updates")
            downloadFiles(updateFiles, isBaseline = false)
        } catch (e: IOException) {
            logger.fatal("Failed to connect to the server. Error message: ${e.printStackTrace()}")
        } finally {
            if (tempDirectory.exists()) {
                logger.info("Deleting directory: ${tempDirectory.absolutePath}")
                tempDirectory.deleteRecursively()
            }
            if (collectStats) {
                logger.info("Writing stats to $statsPath")
                statsPath.toFile().outputStream().bufferedWriter().use {
                    xmlParser.pubmedXMLHandler.tags.iterator().forEach { tag ->
                        it.write("${tag.key} ${tag.value}\n")
                    }
                }
            }
        }
        return true
    }

    private fun unpack(archiveName: String): Boolean {
        val archive = File(archiveName)
        val originalName = archiveName.substringBefore(".gz")
        val bufferSize = 1024
        var safeUnpack = true

        logger.info("$archiveName: Unpacking...")

        GZIPInputStream(BufferedInputStream(FileInputStream(archiveName))).use { inputStream ->
            BufferedOutputStream(FileOutputStream(originalName)).use { outputStream ->
                try {
                    inputStream.copyTo(outputStream, bufferSize)
                } catch (e: EOFException) {
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

    private fun downloadFiles(files: List<String>, isBaseline: Boolean) {
        files.forEach { file ->
            val localArchiveName = "${tempDirectory.absolutePath}/$file"
            val localName = localArchiveName.substringBefore(".gz")

            logger.info("$localArchiveName: Downloading...")

            val downloadSuccess = if (isBaseline)
                ftpHandler.downloadBaselineFile(file, tempDirectory.absolutePath)
            else
                ftpHandler.downloadUpdateFile(file, tempDirectory.absolutePath)
            var overallSuccess = false

            if (downloadSuccess && unpack(localArchiveName)) {
                overallSuccess = xmlParser.parse(localName)
                File(localName).delete()
            }
            // We should save progress information to be able to recover from Ctrl-C/kill signals
            logger.debug("Saving progress to $progressPath")
            BufferedWriter(FileWriter(progressPath.toFile())).use {
                it.write("lastCheck ${Date.from(Instant.now()).time}\n" +
                        "lastId ${PubmedFTPHandler.pubmedFileToId(file)}")
            }

            logger.info("$localName: ${if (overallSuccess) "SUCCESS" else "FAILURE"}")
        }
    }

    private fun loadLastProgress(): Pair<Long?, Int?> {
        var lastCheck: Long? = null
        var lastId: Int? = null
        if (Files.exists(progressPath)) {
            logger.info("Found crawler progress $progressPath")
            BufferedReader(FileReader(progressPath.toFile())).useLines { lines ->
                for (line in lines) {
                    val chunks = line.split(" ")
                    when {
                        chunks.size == 2 && chunks[0] == "lastCheck" -> {
                            lastCheck = chunks[1].toLong()
                            logger.info("lastCheck: $lastCheck")
                        }
                        chunks.size == 2 && chunks[0] == "lastId" -> {
                            lastId = chunks[1].toInt()
                            logger.info("lastId: $lastId")
                        }
                    }
                }
            }
        }
        return lastCheck to lastId
    }

}