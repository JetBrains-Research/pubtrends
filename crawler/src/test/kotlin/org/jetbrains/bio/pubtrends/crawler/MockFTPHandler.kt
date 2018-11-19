package org.jetbrains.bio.pubtrends.crawler

import org.apache.commons.net.ftp.*
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException


class MockFTPHandler {
    companion object {
        const val server = "org/jetbrains/bio/pubtrends/crawler/data/"
        const val baselinePath = "pubmed/baseline/"
        const val updatePath = "pubmed/updatefiles/"
    }

    fun fetch(lastCheck : Long = 0) : Pair<List<String>, List<String>> {
        val ftp = MockFTPClient()

        ftp.use {
            connect(it)
            val baselineFiles = getNewXMLsList(it, baselinePath, lastCheck)
            val updateFiles = getNewXMLsList(it, updatePath, lastCheck)

            return Pair(baselineFiles, updateFiles)
        }
    }

    fun downloadBaselineFile(name : String, localPath: String) : Boolean {
        val ftp = MockFTPClient()

        ftp.use {
            connect(it)

            ftp.changeWorkingDirectory(baselinePath)
            return downloadFile(it, name, localPath)
        }
    }

    fun downloadUpdateFile(name : String, localPath : String) : Boolean {
        val ftp = MockFTPClient()

        ftp.use {
            connect(it)

            ftp.changeWorkingDirectory(updatePath)
            return downloadFile(it, name, localPath)
        }
    }

    private fun connect(ftp : FTPClient) {
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

            if (!ftp.setFileType(FTPClient.BINARY_FILE_TYPE)) {
                throw IOException("Failed to set binary file type.")
            }
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }
    }

    private fun downloadFile(ftp : FTPClient, name : String, localPath : String) : Boolean {
        val localFile = File("$localPath$name")

        BufferedOutputStream(localFile.outputStream()).use {
            try {
                return ftp.retrieveFile(name, it)
            } catch (e: IOException) {
                e.printStackTrace()
                if (ftp.isConnected) {
                    ftp.disconnect()
                }
            }
        }

        return false
    }

    private fun getNewXMLsList(ftp : MockFTPClient, directory : String, lastCheck : Long) : List<String> {
        ftp.changeWorkingDirectory(directory)
        try {
            return ftp.listMockFiles().filter {
                ((it.lastModified() > lastCheck) && (it.name.endsWith(".xml.gz")))
            }.map {
                it.name
            }
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }

        return emptyList()
    }
}