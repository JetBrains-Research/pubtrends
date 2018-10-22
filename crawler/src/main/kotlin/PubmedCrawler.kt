import org.apache.logging.log4j.LogManager
import java.io.*
import java.util.zip.GZIPInputStream

class PubmedCrawler {
    private val logger = LogManager.getLogger(PubmedCrawler::class)
    private val client = PubmedFTPClient()

    fun update() {
        // TODO: to config
        // Timestamp (ms) of 14th Oct 18 ~10:30 to download 8 new XML
        val lastCheck : Long = 1539513468000
        val files = client.fetch(lastCheck)

        val baselineFiles = files.first
        val updateFiles = files.second
        logger.info("Found ${baselineFiles.size + updateFiles.size} new file(s)")
        downloadFiles(baselineFiles, isBaseline = true)
        downloadFiles(updateFiles, isBaseline = false)
    }

    fun unpack(archiveName : String) : Boolean {
        val archive = File("data/$archiveName")
        val originalName = "data/${archiveName.substringBefore(".gz")}"
        val bufferSize = 1024
        var safeUnpack = true

        GZIPInputStream(BufferedInputStream(FileInputStream("data/$archiveName"))).use { inputStream ->
            BufferedOutputStream(FileOutputStream(originalName)).use { outputStream ->
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
            logger.info("$it: Downloading...")

            val downloadSuccess = when (isBaseline) {
                true -> client.downloadBaselineFile(it, "data/")
                false -> client.downloadUpdateFile(it, "data/")
            }

            var overallSuccess = false
            if (downloadSuccess) {
                logger.info("$it: Unpacking...")
                overallSuccess = unpack(it)
            }

            logger.info("$it: ${if (overallSuccess) "SUCCESS" else "FAILURE"}")
        }
    }
}