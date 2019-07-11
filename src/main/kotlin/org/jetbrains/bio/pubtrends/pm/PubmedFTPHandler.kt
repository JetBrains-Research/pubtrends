package org.jetbrains.bio.pubtrends.pm

import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPReply
import org.apache.logging.log4j.LogManager
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException


class PubmedFTPHandler {
    companion object {
        private val logger = LogManager.getLogger(PubmedFTPHandler::class)

        const val server = "ftp.ncbi.nlm.nih.gov"
        const val baselinePath = "/pubmed/baseline"
        const val updatePath = "/pubmed/updatefiles"

        fun pubmedFileToId(name: String): Int = name.removeSurrounding("pubmed19n", ".xml.gz").toInt()

        fun idToPubmedFile(id: Int): String = "pubmed19n${id.toString().padStart(4, '0')}.xml.gz"
    }

    fun fetch(lastId: Int = 0): Pair<List<String>, List<String>> {
        val ftp = CloseableFTPClient()

        ftp.use {
            logger.info("Connecting to $server")
            connect(it)

            logger.info("Fetching baseline files")
            val baselineFiles = getNewXMLsList(it, baselinePath, lastId)
            logger.info("Fetching update files")
            val updateFiles = getNewXMLsList(it, updatePath, lastId)

            return Pair(baselineFiles, updateFiles)
        }
    }

    fun downloadBaselineFile(name: String, localPath: String): Boolean {
        val ftp = CloseableFTPClient()

        ftp.use {
            connect(it)

            ftp.changeWorkingDirectory(baselinePath)
            return downloadFile(it, name, localPath)
        }
    }

    fun downloadUpdateFile(name: String, localPath: String): Boolean {
        val ftp = CloseableFTPClient()

        ftp.use {
            connect(it)

            ftp.changeWorkingDirectory(updatePath)
            return downloadFile(it, name, localPath)
        }
    }

    private fun connect(ftp: FTPClient) {
        try {
            ftp.connect(server)
            val reply = ftp.replyCode

            if (!FTPReply.isPositiveCompletion(reply)) {
                ftp.disconnect()
                throw IOException("FTP server refused connection.")
            }

            ftp.enterLocalPassiveMode()
            if (!ftp.login("anonymous", "")) {
                throw IOException("Failed to log in.")
            }

            ftp.bufferSize = 1024 * 1024
            logger.debug("New buffer size: ${ftp.bufferSize}")

            if (!ftp.setFileType(FTPClient.BINARY_FILE_TYPE)) {
                throw IOException("Failed to set binary file type.")
            }
        } catch (e: IOException) {
            if (ftp.isConnected) {
                ftp.disconnect()
            }
            throw IOException(e)
        }
    }

    private fun downloadFile(ftp: FTPClient, name: String, localPath: String): Boolean {
        val localFile = File("$localPath/$name")

        BufferedOutputStream(localFile.outputStream()).use {
            try {
                return ftp.retrieveFile(name, it)
            } catch (e: IOException) {
                logger.error(e)
                if (ftp.isConnected) {
                    ftp.disconnect()
                }
            }
        }

        return false
    }

    private fun getNewXMLsList(ftp: FTPClient, directory: String, lastId: Int): List<String> {
        ftp.changeWorkingDirectory(directory)
        try {
            return ftp.listFiles()?.filter {
                it.name.endsWith(".xml.gz") && pubmedFileToId(it.name) > lastId
            }?.map {
                it.name
            } ?: emptyList()
        } catch (e: IOException) {
            logger.error(e)
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }

        return emptyList()
    }
}