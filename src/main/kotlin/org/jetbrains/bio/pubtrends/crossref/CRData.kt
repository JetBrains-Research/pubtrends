package com.preprint.server.core.crossref

import com.preprint.server.core.data.Author
import com.preprint.server.core.data.JournalRef
import com.preprint.server.core.data.PubData
import com.preprint.server.core.data.Reference

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