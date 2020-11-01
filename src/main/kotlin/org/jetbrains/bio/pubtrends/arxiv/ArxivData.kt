package org.jetbrains.bio.pubtrends.arxiv

import org.jetbrains.bio.pubtrends.data.Author
import org.jetbrains.bio.pubtrends.data.JournalRef
import org.jetbrains.bio.pubtrends.data.PubData
import org.jetbrains.bio.pubtrends.data.Reference

data class ArxivData(
    val identifier: String = "",
    var datestamp: String = "",
    override var id: String = "",
    override var abstract: String = "",
    var creationDate: String = "",
    override var title: String = "",
    var lastUpdateDate: String = "",
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
