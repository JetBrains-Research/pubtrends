package org.jetbrains.bio.pubtrends.crossref

import org.jetbrains.bio.pubtrends.data.Author
import org.jetbrains.bio.pubtrends.data.JournalRef
import org.jetbrains.bio.pubtrends.data.Reference

data class CRData(
    var id: String = "",
    var title: String = "",
    var doi: String? = null,
    var journal: JournalRef? = null,
    var refList: MutableList<Reference> = mutableListOf(),
    var pdfUrls: MutableList<String> = mutableListOf(),
    val authors: MutableList<Author> = mutableListOf(),
    var abstract: String = ""
)