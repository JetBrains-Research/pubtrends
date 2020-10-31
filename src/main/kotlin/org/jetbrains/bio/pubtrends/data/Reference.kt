package com.preprint.server.core.data

import com.preprint.server.core.ref.GrobidEngine
import com.preprint.server.core.utils.Common
import org.grobid.core.data.BibDataSet
import org.grobid.core.data.BiblioItem


/**
 * Contains information about bibliographic reference
 */
data class Reference(
    var rawReference: String = "",
    var arxivId: String? = null,
    var doi: String? = null,
    var authors: List<Author> = listOf(),
    var title: String? = null,
    var journal: String? = null,
    var issue: String? = null,
    var firstPage: Int? = null,
    var lastPage: Int? = null,
    var volume: String? = null,
    var year: Int? = null,
    var issn: String? = null,
    var pmid: String? = null,
    var ssid: String? = null,
    var urls: MutableList<String> = mutableListOf(),
    var isReference : Boolean = false,
    var validated : Boolean = false
) {

    /**
     * If `shouldParse` == true then Grobid will be used
     * to parse full information from reference.
     * Other constructors use classes from Grobid library to construct reference
     */
    constructor(ref: String, shouldParse: Boolean = false) : this() {
        rawReference = ref
        if (shouldParse) {
            val p = GrobidEngine.processRawReference(rawReference, 1)
            setBib(p)
        }
    }
    constructor(bibData: BibDataSet) : this() {
        val p = bibData.resBib
        rawReference = bibData.rawBib.replace("\n", "")
        setBib(p)
    }
    constructor(rawRef : String, bib : BiblioItem) : this() {
        rawReference = rawRef
        setBib(bib)
    }
    private fun setBib(p: BiblioItem) {
        arxivId = p.arXivId
        doi = p.doi
        authors = p.fullAuthors?.map {author -> Author(author.toString())} ?: listOf()
        title = p.title
        journal = p.journal
        issue = p.issue
        if (!p.pageRange.isNullOrBlank()) {
            val (firstPage_, lastPage_) = Common.splitPages(p.pageRange)
            firstPage = firstPage_
            lastPage = lastPage_
        }
        volume = p.volumeBlock
        year = Common.parseYear(p.publicationDate)
        issn = p.issn
        isReference = !p.rejectAsReference()
    }

    override fun toString() : String {
        var res = "record\n"
        fun addField(field: String, value : String?) {
            if (value != null) {
                res += "$field: $value\n"
            }
        }
        res += "\n"
        addField("raw:", rawReference)
        addField("    title", title)
        addField("    authors", authors.joinToString { it.toString() })
        addField("    arxiv id", arxivId)
        addField("    doi", doi)
        addField("    journal", journal)
        addField("    volume", volume)
        addField("    issue", issue)
        addField("    year", "$year")
        addField("    pages", "$firstPage-$lastPage")
        addField("    validated", validated.toString())
        res += "\n\n"
        return res
    }

    companion object {
        fun toReferences(refList: List<String>): List<Reference> {
            val p = GrobidEngine.processRawReferences(refList, 0)
            return refList.zip(p).map { (ref, bib) -> Reference(ref, bib) }
        }
    }
}