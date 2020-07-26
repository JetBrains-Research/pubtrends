import com.preprint.server.core.arxiv.ArxivCollector
import com.preprint.server.core.neo4j.DatabaseHandler
import com.preprint.server.core.validation.ArxivValidator
import com.preprint.server.core.validation.CrossRefValidator
import com.preprint.server.core.validation.LocalValidator
import joptsimple.OptionParser
import java.io.IOException
import java.lang.IllegalArgumentException

fun main(args: Array<String>) {
    val optionParser = OptionParser()
    optionParser.accepts("from").withRequiredArg()
    optionParser.accepts("validators").withRequiredArg()

    val options = optionParser.parse(*args)

    val startDate = if (options.has("from")) {
        options.valueOf("from").toString()
    } else {
        throw IllegalArgumentException("You should specify `from` option`")
    }

    var validators = mutableListOf(CrossRefValidator, ArxivValidator)
    if (options.has("validators")) {
        validators = mutableListOf()
        options.valueOf("validators").toString().forEach { c ->
            when (c) {
                'l' -> validators.add(LocalValidator)
                'c' -> validators.add(CrossRefValidator)
                'a' -> validators.add(ArxivValidator)
                else -> throw IOException("Wrong argument for --validators")
            }
        }
    }

    println("ArxivCollector will be launched with the following parameters")
    println("start date: $startDate")
    println("validators: ${validators.joinToString(separator = ", ") {it.javaClass.name}}")

    val dbHandler = DatabaseHandler(
        Config.config["neo4j_url"].toString(),
        Config.config["neo4j_port"].toString(),
        Config.config["neo4j_user"].toString(),
        Config.config["neo4j_password"].toString()
    )

    ArxivCollector.collect(startDate, dbHandler, validators)
}