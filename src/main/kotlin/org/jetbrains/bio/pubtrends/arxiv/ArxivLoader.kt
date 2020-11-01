package org.jetbrains.bio.pubtrends.arxiv

import org.jetbrains.bio.pubtrends.db.ArxivNeo4JWriter
import org.jetbrains.bio.pubtrends.validation.ArxivValidator
import org.jetbrains.bio.pubtrends.validation.CrossRefValidator
// TODO(kapralov): local validator is an external dependency, not yet sure whether it is needed
//import org.jetbrains.bio.pubtrends.validation.LocalValidator
import joptsimple.OptionParser
import org.apache.logging.log4j.LogManager
import org.jetbrains.bio.pubtrends.Config
import java.io.IOException
import java.lang.IllegalArgumentException

object ArxivLoader {
    @JvmStatic
    fun main(args: Array<String>) {
        val logger = LogManager.getLogger("Pubtrends")

        with (OptionParser()) {
            accepts("from").withRequiredArg()
            accepts("validators").withRequiredArg()

            val options = parse(*args)

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
                        // TODO(kapralov): local validator is an external dependency, not yet sure whether it is needed
                        //                'l' -> validators.add(LocalValidator)
                        'c' -> validators.add(CrossRefValidator)
                        'a' -> validators.add(ArxivValidator)
                        else -> throw IOException("Wrong argument for --validators")
                    }
                }
            }

            logger.info("ArxivCollector will be launched with the following parameters")
            logger.info("start date: $startDate")
            logger.info("validators: ${validators.joinToString(separator = ", ") { it.javaClass.name }}")

            val dbHandler = ArxivNeo4JWriter(
                    Config.config["neo4j_host"].toString(),
                    Config.config["neo4j_port"].toString(),
                    Config.config["neo4j_username"].toString(),
                    Config.config["neo4j_password"].toString()
            )

            ArxivCollector.collect(startDate, dbHandler, validators)
        }
    }
}