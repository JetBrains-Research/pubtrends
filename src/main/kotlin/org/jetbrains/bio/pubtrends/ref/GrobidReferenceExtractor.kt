package com.preprint.server.core.ref

import com.preprint.server.core.data.Reference
import java.io.File


object GrobidReferenceExtractor : ReferenceExtractor {
    override fun extractUnverifiedReferences(pdf: ByteArray): List<Reference> {
        val tmpFile = createTempFile()
        tmpFile.writeBytes(pdf)
        val res = GrobidEngine.processReferences(tmpFile, 0)
        tmpFile.deleteOnExit()
        return res.map { Reference(it) }
    }
}