package preprint.server.examples

import org.jetbrains.bio.pubtrends.data.Reference
import org.jetbrains.bio.pubtrends.ref.CustomReferenceExtractor
import org.jetbrains.bio.pubtrends.ref.GrobidReferenceExtractor
import org.jetbrains.bio.pubtrends.validation.ArxivValidator
import org.jetbrains.bio.pubtrends.validation.LocalValidator
import java.io.File
import java.nio.file.Paths
import kotlin.system.measureTimeMillis

const val FILES_FOLDER = "./files/test1/"
const val BENCHMARKS_FOLDER = "./benchmarks/data"

/**
 * Compare implemented reference extractors on a set of files from FILES_FOLDER.
 * Output CSV contains the following metrics:
 *  - number of parsed references
 *  - execution time in milliseconds
 * Format: filename,extractor,referencesNumber,executionTime
 *
 * This CSV is used by benchmark.ipynb to summarize and visualize results.
 */
fun main() {
    System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog")

    val validators = listOf(LocalValidator, ArxivValidator)

    val extractors = mapOf(
        CustomReferenceExtractor to "Custom",
        GrobidReferenceExtractor to "Grobid"
    )

    val references = mutableMapOf(
        CustomReferenceExtractor to mutableListOf<Pair<String, Reference>>(),
        GrobidReferenceExtractor to mutableListOf<Pair<String, Reference>>()
    )

    val pdfPath = Paths.get(FILES_FOLDER)
    File(BENCHMARKS_FOLDER).mkdir()
    val csvStatsFile = File(BENCHMARKS_FOLDER, "benchmark.csv")
    csvStatsFile.outputStream().bufferedWriter().use { csvWriter ->
        pdfPath.toAbsolutePath().normalize().toFile().listFiles { file ->
            file.name.endsWith(".pdf")
        }?.let { files ->
            files.forEachIndexed { i, file ->
                extractors.forEach { extractor, extName ->
                    var referencesNumber = 0
                    val progressPrefix = "(${i + 1} / ${files.size})"
                    var newRefs = listOf<Reference>()
                    val timeMillis = measureTimeMillis {
                        try {
                            newRefs = extractor.getReferences(file.readBytes())
                            references[extractor]!!.addAll(newRefs.map {Pair(file.nameWithoutExtension, it)})
                            referencesNumber = newRefs.size
                        } catch (e: Exception) {
                            println("$progressPrefix ${file.nameWithoutExtension},${extractors[extractor]} - e.message")
                        }
                    }
                    if (i > 0) {
                        validators.forEach { it.validate(newRefs) }
                        val validatedNumber = newRefs.count { it.validated }
                        println("Validated $validatedNumber out of ${newRefs.size} ($extName)")
                        val data =
                            "${file.nameWithoutExtension},${extractors[extractor]}," +
                                    "${referencesNumber},${validatedNumber},${timeMillis}"
                        csvWriter.write("$data\n")
                        println("$progressPrefix $data")
                    }
                }
            }
        }
    }

    /**
     * Crete files that stores differences in extracted references(only validated are considered)
     * For example reference that was extracted with Grobid and was not extracted by Custom
     * will be stored in file GrobidWithoutCustom.txt.
     * This will be done for each file separately and references will be compared by doi or arxivId
     */

    val pairwiseBenchmark = File(BENCHMARKS_FOLDER, "pairwise.csv")
    pairwiseBenchmark.writeText(extractors.values.joinToString(separator = ","))
    for ((extractor1, name1) in extractors) {
        for ((extractor2, name2) in extractors) {
            if (extractor1 == extractor2) {
                continue
            }
            val m1 = references[extractor1]!!.groupBy({it.first}, {it.second})
            val m2 = references[extractor2]!!.groupBy({it.first}, {it.second})
            val outputFile = File(BENCHMARKS_FOLDER, "${name1}Without${name2}")
            var diffCnt = 0
            outputFile.writeText("")
            for ((filename, refs) in m1) {
                val valRefs1 = refs.filter {it.validated}
                val valRefs2 = m2[filename]?.filter { it.validated }
                if (valRefs2 == null) {
                    valRefs1.forEach { outputFile.appendText("$filename\n $it\n\n") }
                    diffCnt += valRefs1.size
                } else {
                    for (ref in valRefs1) {
                        if (
                            valRefs2.count {
                                ref.arxivId != null && ref.arxivId == it.arxivId
                                    || ref.doi != null && ref.doi == it.doi } == 0
                        ) {
                            outputFile.appendText("$filename\n $ref\n\n")
                            diffCnt += 1
                        }
                    }
                }
            }
            pairwiseBenchmark.appendText("\n$name1,$name2,$diffCnt")
            println("$name1 extractor extracted $diffCnt references more than $name2")
        }
    }
}