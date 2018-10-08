import org.apache.commons.net.ftp.FTPClient
import org.apache.commons.net.ftp.FTPFile
import org.apache.commons.net.ftp.FTPReply
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException
import java.io.OutputStream

class PubmedFTPClient {
    private val ftp = FTPClient()
    private val server = "ftp.ncbi.nlm.nih.gov"
    private val path = "/pubmed/updatefiles/"

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

            ftp.changeWorkingDirectory(path)
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }
    }

    fun downloadFile(name : String) : Boolean {
        try {
            return ftp.retrieveFile(name, BufferedOutputStream(File(name).outputStream()))
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }
        return false
    }

    fun getNewXMLsList(lastCheck : Long) : List<FTPFile>? {
        try {
            val files = ftp.listFiles()
            return files?.filter {
                ((it.timestamp.time.time > lastCheck) && (it.name.endsWith(".xml.gz")))
            }
        } catch (e: IOException) {
            e.printStackTrace()
            if (ftp.isConnected) {
                ftp.disconnect()
            }
        }

        return null
    }

    private fun finalize() {
        if (ftp.isConnected) {
            ftp.disconnect()
        }
    }
}