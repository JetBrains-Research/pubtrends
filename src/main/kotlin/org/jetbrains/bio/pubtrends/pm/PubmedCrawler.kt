package org.jetbrains.bio.pubtrends.pm

import org.apache.logging.log4j.LogManager
import java.io.*
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.GZIPInputStream

class PubmedCrawler(
    private val xmlParser: PubmedXMLParser,
    private val collectStats: Boolean,
    private val statsTSV: Path,
    private val progressTSV: Path
) {

    private val logger = LogManager.getLogger(PubmedCrawler::class)

    init {
        if (collectStats) {
            logger.info("Collecting stats in $statsTSV")
        }
    }

    private val ftpHandler = PubmedFTPHandler()
    private lateinit var tempDirectory: File

    /**
     * @return false if update not required
     */
    fun update(lastIdCmd: Int?): Boolean {
        var lastId = 0
        if (lastIdCmd == null) {
            if (Files.exists(progressTSV)) {
                logger.info("Found crawler progress $progressTSV")
                BufferedReader(FileReader(progressTSV.toFile())).useLines { lines ->
                    for (line in lines) {
                        val chunks = line.split("\t")
                        when {
                            chunks.size == 2 && chunks[0] == "lastId" -> {
                                lastId = chunks[1].toInt()
                            }
                        }
                    }
                }
            }
        } else {
            lastId = lastIdCmd
        }

        if (lastId > 0) {
            logger.info("Last downloaded file: ${PubmedFTPHandler.idToPubmedFile(lastId)}")
        }

        tempDirectory = createTempDir()
        logger.info("Created temporary directory: ${tempDirectory.absolutePath}")
        try {
            val (baselineFiles, updateFiles) = ftpHandler.fetch(lastId)
            val baselineSize = baselineFiles.size
            val updatesSize = updateFiles.size
            logger.info(
                "Found ${baselineSize + updatesSize} new file(s)\n" +
                        "Baseline: $baselineSize, Updates: $updatesSize"
            )
            if (baselineSize + updatesSize == 0) {
                return false
            }
            logger.info("Processing baseline")
            downloadFiles(baselineFiles, isBaseline = true)
            logger.info("Processing updates")
            downloadFiles(updateFiles, isBaseline = false)
        } catch (e: IOException) {
            logger.error("Failed to connect to the server", e)
        } finally {
            if (tempDirectory.exists()) {
                logger.info("Deleting directory: ${tempDirectory.absolutePath}")
                tempDirectory.deleteRecursively()
            }
            if (collectStats) {
                logger.info("Writing stats to $statsTSV")
                statsTSV.toFile().outputStream().bufferedWriter().use {
                    xmlParser.tags.iterator().forEach { tag ->
                        it.write("${tag.key}\t${tag.value}\n")
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

        GZIPInputStream(BufferedInputStream(FileInputStream(archiveName))).use { inputStream ->
            BufferedOutputStream(FileOutputStream(originalName)).use { outputStream ->
                try {
                    inputStream.copyTo(outputStream, bufferSize)
                } catch (e: EOFException) {
                    logger.error("Corrupted GZ archive. ", e)
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
        val filesSize = files.size
        val fileType = if (isBaseline) "baseline" else "update"

        files.forEachIndexed { idx, file ->
            val localArchiveName = "${tempDirectory.absolutePath}/$file"
            val localName = localArchiveName.substringBefore(".gz")
            val progressPrefix = "(${idx + 1} / $filesSize $fileType)"

            logger.info("$progressPrefix $localArchiveName: Downloading...")

            val downloadSuccess = if (isBaseline)
                ftpHandler.downloadBaselineFile(file, tempDirectory.absolutePath)
            else
                ftpHandler.downloadUpdateFile(file, tempDirectory.absolutePath)
            var overallSuccess = false

            logger.info("$progressPrefix $localArchiveName: Unpacking...")
            if (downloadSuccess && unpack(localArchiveName)) {
                logger.info("$progressPrefix $localName: Parsing...")
                overallSuccess = xmlParser.parse(localName)
                File(localName).delete()
            }

            logger.info("$progressPrefix $localName: ${if (overallSuccess) "SUCCESS" else "FAILURE"}")

            // Save progress information to be able to recover from Ctrl-C/kill signals
            logger.debug("$progressPrefix Save progress to $progressTSV")
            BufferedWriter(FileWriter(progressTSV.toFile())).use {
                it.write("lastId\t${PubmedFTPHandler.pubmedFileToId(file)}")
            }
        }
    }
}