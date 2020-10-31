package org.jetbrains.bio.pubtrends.ref

import org.jetbrains.bio.pubtrends.data.Reference
import pl.edu.icm.cermine.ContentExtractor
import java.io.ByteArrayInputStream
import java.io.InputStream

object CermineReferenceExtractor : ReferenceExtractor {
    override fun extractUnverifiedReferences(pdf: ByteArray): List<Reference> {
        val extractor = ContentExtractor()
        val inputStream: InputStream = ByteArrayInputStream(pdf)
        extractor.setPDF(inputStream)
        val references = extractor.references
        return Reference.toReferences(references.map { it.text })
    }
}