package org.jetbrains.bio.pubtrends.pm

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPReply
import org.slf4j.LoggerFactory
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException


class PubmedFTPHandler {
    companion object {
        private val LOG = LoggerFactory.getLogger(PubmedFTPHandler::class.java)

        const val SERVER = "ftp.ncbi.nlm.nih.gov"
        const val BASELINE_PATH = "/pubmed/baseline"
        const val UPDATE_PATH = "/pubmed/updatefiles"

        const val TIMEOUT_MS = 20000
        const val DOWNLOAD_TIMEOUT_MS = 100000L
        const val YEAR = 25

        fun pubmedFileToId(name: String): Int = name.removeSurrounding("pubmed${YEAR}n", ".xml.gz").toInt()

        fun idToPubmedFile(id: Int): String = "pubmed${YEAR}n${id.toString().padStart(4, '0')}.xml.gz"
    }

    fun fetch(lastId: Int = 0): Pair<List<String>, List<String>> {
        val ftp = CloseableFTPClient()

        ftp.use {
            LOG.info("Connecting to $SERVER")
            connect(it)

            LOG.info("Fetching baseline files")
            val baselineFiles = getNewXMLsList(it, BASELINE_PATH, lastId)
            LOG.info("Fetching update files")
            val updateFiles = getNewXMLsList(it, UPDATE_PATH, lastId)

            return Pair(baselineFiles, updateFiles)
        }
    }

    fun downloadBaselineFile(name: String, localPath: String) {
        val ftp = CloseableFTPClient()

        ftp.use {
            connect(it)

            ftp.changeWorkingDirectory(BASELINE_PATH)
            downloadFile(it, name, localPath)
        }
    }

    fun downloadUpdateFile(name: String, localPath: String) {
        val ftp = CloseableFTPClient()

        ftp.use {
            connect(it)

            ftp.changeWorkingDirectory(UPDATE_PATH)
            downloadFile(it, name, localPath)
        }
    }

    @Throws(IOException::class)
    private fun connect(ftp: FTPClient) {
        ftp.connect(SERVER)
        val reply = ftp.replyCode

        if (!FTPReply.isPositiveCompletion(reply)) {
            ftp.disconnect()
            throw IOException("FTP server refused connection.")
        }

        ftp.enterLocalPassiveMode()
        if (!ftp.login("anonymous", "")) {
            throw IOException("Failed to log in.")
        }

        // Timeouts are set to avoid infinite download
        ftp.defaultTimeout = TIMEOUT_MS
        ftp.setDataTimeout(TIMEOUT_MS)
        ftp.connectTimeout = TIMEOUT_MS
        ftp.soTimeout = TIMEOUT_MS
        ftp.controlKeepAliveTimeout = TIMEOUT_MS.toLong()
        ftp.controlKeepAliveReplyTimeout = TIMEOUT_MS

        if (!ftp.setFileType(FTPClient.BINARY_FILE_TYPE)) {
            throw IOException("Failed to set binary file type.")
        }
    }

    private fun downloadFile(ftp: FTPClient, name: String, localPath: String) {
        val localFile = File("$localPath/$name")

        runBlocking {
            withTimeout(timeMillis = DOWNLOAD_TIMEOUT_MS) {
                BufferedOutputStream(localFile.outputStream()).use {
                    ftp.retrieveFile(name, it)
                }
            }
        }
    }

    private fun getNewXMLsList(ftp: FTPClient, directory: String, lastId: Int): List<String> {
        ftp.changeWorkingDirectory(directory)

        return ftp.listFiles()?.filter {
            it.name.endsWith(".xml.gz") && pubmedFileToId(it.name) > lastId
        }?.map {
            it.name
        } ?: emptyList()
    }
}