import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException

import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPFile
import org.apache.commons.net.ftp.FTPReply


class PubmedFTPClient {
    private val ftp = FTPClient()
    private val server = "ftp.ncbi.nlm.nih.gov"
    private val baselinePath = "/pubmed/baseline/"
    private val updatePath = "/pubmed/updatefiles/"

    init {
        try {
            ftp.connect(server)
            val reply = ftp.replyCode

            if (!FTPReply.isPositiveCompletion(reply)) {
                ftp.disconnect()
                throw Exception("FTP server refused connection.")
            }

            ftp.enterLocalPassiveMode()
            if (!ftp.login("anonymous", "")) {
                throw Exception("Failed to log in.")
            }

            if (!ftp.setFileType(FTPClient.BINARY_FILE_TYPE)) {
                throw Exception("Failed to set binary file type.")
            }
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }
    }

    fun downloadFile(name : String, localPath : String) : Boolean {
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

    fun downloadBaselineFile(name: String, localPath: String) : Boolean {
        if (ftp.printWorkingDirectory() != baselinePath) {
            ftp.changeWorkingDirectory(baselinePath)
        }
        return downloadFile(name, localPath)
    }

    fun downloadUpdateFile(name: String, localPath: String) : Boolean {
        if (ftp.printWorkingDirectory() != updatePath) {
            ftp.changeWorkingDirectory(updatePath)
        }
        return downloadFile(name, localPath)
    }

    fun getBaselineXMLsList() : List<FTPFile>? {
        ftp.changeWorkingDirectory(baselinePath)
        try {
            return ftp.listFiles()?.filter {
                (it.name.endsWith(".xml.gz"))
            }
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }

        return emptyList()
    }

    fun getNewXMLsList(lastCheck : Long) : List<FTPFile>? {
        ftp.changeWorkingDirectory(updatePath)
        try {
            return ftp.listFiles()?.filter {
                ((it.timestamp.time.time > lastCheck) && (it.name.endsWith(".xml.gz")))
            }
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }

        return emptyList()
    }

    private fun finalize() {
        if (ftp.isConnected) {
            ftp.disconnect()
        }
    }
}