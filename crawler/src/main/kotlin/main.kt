fun main(args: Array<String>) {
    // TODO: to config
    // Timestamp of 7th Oct 18 ~16:00 to download 1 new XML
    val lastCheck : Long = 1538917109000

    val client = PubmedFTPClient()
    val files = client.getNewXMLsList(lastCheck)
    if (files != null) {
        println("Found ${files.size} new file(s)")
        for (file in files) {
            if (client.downloadFile(file.name)) {
                println("${file.name}: SUCCESS")
            } else {
                println("${file.name}: FAILURE")
            }
        }
    }
}