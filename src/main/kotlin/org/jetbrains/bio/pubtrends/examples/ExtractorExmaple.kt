package examples

import org.jetbrains.bio.pubtrends.ref.CustomReferenceExtractor
import java.io.File

fun main() {
    val bytes = File("/home/ilya/Downloads/a.pdf").readBytes()
    CustomReferenceExtractor.extractUnverifiedReferences(bytes).forEach {
        println(it)
    }

}