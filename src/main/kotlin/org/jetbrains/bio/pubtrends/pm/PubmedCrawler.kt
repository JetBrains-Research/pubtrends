package org.jetbrains.bio.pubtrends.pm

import kotlinx.coroutines.TimeoutCancellationException
import org.apache.logging.log4j.LogManager
import java.io.*
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.GZIPInputStream
import javax.xml.stream.XMLStreamException

class PubmedCrawlerException : Exception {
    constructor(message: String) : super(message)
    constructor(cause: Throwable) : super(cause)
    constructor(message: String, cause: Throwable) : super(message, cause)
}

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

        try {
            tempDirectory = createTempDir()
        } catch (e: IOException) {
            throw PubmedCrawlerException("Failed to create temporary directory")
        }
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
            logger.error("Download failed: ${e.message}")
            throw PubmedCrawlerException(e)
        } catch (e: TimeoutCancellationException) {
            logger.error("Download timed out")
            throw PubmedCrawlerException(e)
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

        return false
    }

    private fun downloadFiles(files: List<String>, isBaseline: Boolean) {
        val filesSize = files.size
        val fileType = if (isBaseline) "baseline" else "update"

        files.forEachIndexed { idx, file ->
            val localArchiveName = "${tempDirectory.absolutePath}/$file"
            val progressPrefix = "(${idx + 1} / $filesSize $fileType)"

            logger.info("$progressPrefix $localArchiveName: Downloading...")

            try {
                if (isBaseline)
                    ftpHandler.downloadBaselineFile(file, tempDirectory.absolutePath)
                else
                    ftpHandler.downloadUpdateFile(file, tempDirectory.absolutePath)
            } catch (e: IOException) {
                deleteIfExists(localArchiveName)
                throw PubmedCrawlerException("Failed to download XML archive", e)
            }

            try {
                logger.info("$progressPrefix $localArchiveName: Parsing...")
                xmlParser.parse(localArchiveName)
            } catch (e: XMLStreamException) {
                throw PubmedCrawlerException("Failed to parse $localArchiveName", e)
            } finally {
                deleteIfExists(localArchiveName)
            }

            logger.info("$progressPrefix $localArchiveName: SUCCESS")

            // Save progress information to be able to recover from Ctrl-C/kill signals
            logger.debug("$progressPrefix Save progress to $progressTSV")
            BufferedWriter(FileWriter(progressTSV.toFile())).use {
                it.write("lastId\t${PubmedFTPHandler.pubmedFileToId(file)}")
            }
        }
    }

    private fun deleteIfExists(name: String) {
        val file = File(name)
        if (file.exists()) {
            file.delete()
        }
    }
}