package com.preprint.server.core.arxiv

import com.preprint.server.core.data.Author
import com.preprint.server.core.data.JournalRef
import com.preprint.server.core.data.PubData
import com.preprint.server.core.data.Reference

data class ArxivData(
    val identifier: String = "",
    var datestamp: String = "",
    override var id: String = "",
    override var abstract: String = "",
    var creationDate: String = "",
    override var title: String = "",
    var lastUpdateDate: String? = null,
    override val authors: MutableList<Author> = mutableListOf(),
    var categories: MutableList<String> = mutableListOf(),
    var comments: String? = null,
    var reportNo: String? = null,
    override var journal: JournalRef? = null,
    var mscClass: String? = null,
    var acmClass: String? = null,
    override var doi: String? = null,
    var license: String? = null,
    override var refList: MutableList<Reference> = mutableListOf(),
    override var pdfUrl: String = ""
): PubData
