package org.jetbrains.bio.pubtrends.ref

import org.apache.logging.log4j.kotlin.logger
import org.grobid.core.data.BibDataSet
import org.grobid.core.data.BiblioItem
import org.grobid.core.engines.Engine
import org.grobid.core.factory.GrobidFactory
import org.grobid.core.main.GrobidHomeFinder
import org.grobid.core.utilities.GrobidProperties
import org.jetbrains.bio.pubtrends.Config
import java.io.File
import java.util.*

object GrobidEngine {
    private val logger = logger()
    var engine : Engine
    init {
        //the path to the grobid home folder
        val homePath = Config.config["grobid_home"].toString()
        val grobidHomeFinder = GrobidHomeFinder(Arrays.asList(homePath))
        GrobidProperties.getInstance(grobidHomeFinder)

        engine = GrobidFactory.getInstance().createEngine()
    }

    fun processReferences(pdfFile : File, consolidate : Int) : List<BibDataSet> {
        logger.debug("Begin process references")
        var tries = 0
        while (true) {
            try {
                return engine.processReferences(pdfFile, consolidate)
            } catch (e: Exception) {
                tries += 1
                if (tries == 3) {
                    throw e
                }
            }
        }
    }

    fun processRawReference(ref : String, consolidate: Int) : BiblioItem {
        var tries = 0
        while (true) {
            try {
                while (true) {
                    val bibRef = engine.processRawReference(ref, consolidate)
                    if (bibRef != null) {
                        return bibRef
                    }
                    tries += 1
                    if (tries == 4) {
                        break
                    }
                }
                return BiblioItem()
            } catch (e: Exception) {
                tries += 1
                if (tries == 4) {
                    return BiblioItem()
                }
            }
        }
    }

    fun processRawReferences(refList : List<String>, consolidate: Int) : List<BiblioItem> {
        logger.debug("Begin process raw references")
        var tries = 0
        while (true) {
            try {
                if (consolidate == 1) {
                    return engine.processRawReferences(refList, consolidate)
                } else {
                    return refList.map { processRawReference(it, 0) }
                }
            } catch (e: Exception) {
                tries += 1
                if (tries == 3) {
                    throw e
                }
            }
        }
    }
}