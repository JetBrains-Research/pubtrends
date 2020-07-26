package examples

import com.preprint.server.core.ref.CustomReferenceExtractor
import java.io.File

fun main() {
    val bytes = File("/home/ilya/Downloads/a.pdf").readBytes()
    CustomReferenceExtractor.extractUnverifiedReferences(bytes).forEach {
        println(it)
    }

}