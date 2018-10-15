import java.io.*
import java.util.zip.GZIPInputStream


fun download(name : String) {
    val client = PubmedFTPClient()
    if (client.downloadFile(name)) {
        println("$name: SUCCESS")
    } else {
        println("$name: FAILURE")
    }
}

fun update() {
    // TODO: to config
    // Timestamp (ms) of 14th Oct 18 ~10:30 to download 1 new XML
    val lastCheck : Long = 1539513468000

    val client = PubmedFTPClient()
    val files = client.getNewXMLsList(lastCheck)
    if (files != null) {
        println("Found ${files.size} new file(s)")
        for (file in files) {
            print("${file.name}: Downloading... ")
            if (client.downloadFile(file.name)) {
                print("Unpacking... ")
                if (unpack(file.name)) {
                    println("SUCCESS")
                } else {
                    println("FAILURE")
                }
            } else {
                println("${file.name}: FAILURE")
            }
        }
    }
}

fun unpack(archiveName : String) : Boolean {
    // Removing '.gz' ending of length 3
    val archive = File(archiveName)
    val originalName = archiveName.dropLast(3)
    val bufferSize = 1024
    var safeUnpack = true

    GZIPInputStream(BufferedInputStream(FileInputStream(archiveName))).use { inputStream ->
        BufferedOutputStream(FileOutputStream(originalName)).use { outputStream ->
            try {
                inputStream.copyTo(outputStream, bufferSize)
            } catch (e : EOFException) {
                print("Warning: Corrupted GZ archive. ")
                safeUnpack = false
            }
        }
    }

    if (safeUnpack) {
        archive.delete()
    }

    return safeUnpack
}

fun main(args: Array<String>) {
    update()
}