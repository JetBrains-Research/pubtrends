import java.io.BufferedReader
import java.io.FileReader
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*

internal object Config {
    // Configure settings folder
    private val settingsRoot: Path = Paths.get(System.getProperty("user.home", ""), ".preprint_server")

    init {
        check(Files.exists(settingsRoot)) {
            "$settingsRoot should have been created by log4j"
        }
    }

    private val configPath: Path = settingsRoot.resolve("config.properties")

    init {
        check(Files.exists(configPath)) {
            "Config file not found, please modify and copy config.properties to $configPath"
        }
    }

    val config by lazy {
        Properties().apply {
            load(BufferedReader(FileReader(configPath.toFile())))
        }
    }
}