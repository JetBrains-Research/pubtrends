package org.jetbrains.bio.pubtrends.crawler

import org.apache.commons.net.ftp.FTPClient
import org.apache.logging.log4j.LogManager
import java.io.Closeable
import java.io.File
import java.io.OutputStream

class MockFTPClient : FTPClient(), Closeable {
    private val logger = LogManager.getLogger(MockFTPClient::class)
    private var origin = ""
    private var current = origin

    init {
        logger.debug("Origin: ${File(origin).canonicalPath}")
    }

    override fun changeWorkingDirectory(pathname: String?): Boolean {
        current = origin + (pathname ?: "")
        logger.debug("Current: ${File(current).canonicalPath}")
        return true
    }

    override fun connect(path: String) {
        origin += path
        _replyCode = 200
        logger.debug("Origin: ${File(origin).canonicalPath}")
    }

    override fun disconnect() {
        close()
    }

    override fun enterLocalPassiveMode() {}

    fun listMockFiles(): Array<File> {
        return File(current).listFiles()
    }

    override fun login(username: String?, password: String?): Boolean {
        return true
    }

    override fun retrieveFile(remote: String?, local: OutputStream?): Boolean {
        if ((remote != null) && (local != null)) {
            File(remote).inputStream().copyTo(local)
        }
        return true
    }

    override fun setFileType(fileType: Int): Boolean {
        return true
    }

    override fun close() {
        logger.debug("Connection was closed.")
    }
}