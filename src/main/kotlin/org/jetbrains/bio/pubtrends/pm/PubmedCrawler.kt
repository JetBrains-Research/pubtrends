package org.jetbrains.bio.pubtrends.pm

import kotlinx.coroutines.TimeoutCancellationException
import org.slf4j.LoggerFactory
import java.io.*
import java.nio.file.Files
import java.nio.file.Path
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

    init {
        if (collectStats) {
            LOG.info("Collecting stats in $statsTSV")
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
                LOG.info("Found crawler progress $progressTSV")
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
            LOG.info("Last downloaded file: ${PubmedFTPHandler.idToPubmedFile(lastId)}")
        }

        try {
            tempDirectory = createTempDir()
        } catch (e: IOException) {
            throw PubmedCrawlerException("Failed to create temporary directory")
        }
        LOG.info("Created temporary directory: ${tempDirectory.absolutePath}")

        try {
            val (baselineFiles, updateFiles) = ftpHandler.fetch(lastId)
            val baselineSize = baselineFiles.size
            val updatesSize = updateFiles.size
            val totalSize = baselineSize + updatesSize
            LOG.info(
                    "Found $totalSize new file(s)\nBaseline: $baselineSize, Updates: $updatesSize"
            )
            if (baselineSize + updatesSize == 0) {
                return false
            }
            LOG.info("Processing baseline")
            downloadAndProcessFiles(baselineFiles, 0, totalSize, isBaseline = true)
            LOG.info("Processing updates")
            downloadAndProcessFiles(updateFiles, baselineSize, totalSize, isBaseline = false)
        } catch (e: IOException) {
            LOG.error("Download failed: ${e.message}")
            throw PubmedCrawlerException(e)
        } catch (e: TimeoutCancellationException) {
            LOG.error("Download timed out")
            throw PubmedCrawlerException(e)
        } finally {
            if (tempDirectory.exists()) {
                LOG.info("Deleting directory: ${tempDirectory.absolutePath}")
                tempDirectory.deleteRecursively()
            }
            if (collectStats) {
                LOG.info("Writing stats to $statsTSV")
                statsTSV.toFile().outputStream().bufferedWriter().use {
                    xmlParser.tags.iterator().forEach { tag ->
                        it.write("${tag.key}\t${tag.value}\n")
                    }
                }
            }
        }

        return false
    }

    private fun downloadAndProcessFiles(
            files: List<String>,
            startProgress: Int,
            totalProgress: Int,
            isBaseline: Boolean
    ) {
        val fileType = if (isBaseline) "baseline" else "update"

        files.forEachIndexed { idx, file ->
            val localArchiveName = "${tempDirectory.absolutePath}/$file"
            val progressPrefix = "(${startProgress + idx + 1} / $totalProgress total) [$fileType]"

            LOG.info("$progressPrefix $localArchiveName: Downloading...")

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
                LOG.info("$progressPrefix $localArchiveName: Parsing...")
                xmlParser.parse(localArchiveName)
            } catch (e: XMLStreamException) {
                throw PubmedCrawlerException("Failed to parse $localArchiveName", e)
            } finally {
                deleteIfExists(localArchiveName)
            }

            LOG.info("$progressPrefix $localArchiveName: SUCCESS")

            // Save progress information to be able to recover from Ctrl-C/kill signals
            LOG.debug("$progressPrefix Save progress to $progressTSV")
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

    companion object {
        private val LOG = LoggerFactory.getLogger(PubmedCrawler::class.java)
    }
}