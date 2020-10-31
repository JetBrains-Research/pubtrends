package org.jetbrains.bio.pubtrends.ref

import org.jetbrains.bio.pubtrends.data.Reference


object GrobidReferenceExtractor : ReferenceExtractor {
    override fun extractUnverifiedReferences(pdf: ByteArray): List<Reference> {
        val tmpFile = createTempFile()
        tmpFile.writeBytes(pdf)
        val res = GrobidEngine.processReferences(tmpFile, 0)
        tmpFile.deleteOnExit()
        return res.map { Reference(it) }
    }
}