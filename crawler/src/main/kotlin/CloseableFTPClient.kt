import org.apache.commons.net.ftp.FTPClient
import java.io.Closeable

class CloseableFTPClient : FTPClient(), Closeable {
    override fun close() {
        if (isConnected) {
            disconnect()
        }
    }
}