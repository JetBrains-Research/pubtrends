package com.preprint.server.core.data

import com.preprint.server.core.ref.GrobidEngine
import com.preprint.server.core.utils.Common
import org.grobid.core.data.BiblioItem

/**
 * Contains information about journal reference.
 * Sometimes may be used to store information about journal
 */
data class JournalRef(
    var rawRef: String,
    var rawTitle: String? = null,
    var volume: String? = null,
    var firstPage: Int? = null,
    var lastPage: Int? = null,
    var year: Int? = null,
    var number: String? = null,
    var issn: String? = null,
    var shortTitle: String? = null,
    var fullTitle: String? = null
) {

    /**
     * If `parse` == true, then Grobid will be used to parse
     * information from `rawRef`
     */
    constructor(rawRef: String, parse: Boolean) : this(rawRef) {
        if (parse) {
            getFullJournalInfo(this)
        }
    }

    constructor(bib: BiblioItem, rawRef: String) : this(rawRef) {
        rawTitle = bib.journal
        val (fp, lp) = Common.splitPages(bib.pageRange)
        firstPage = fp
        lastPage = lp
        volume = bib.volumeBlock
        year = Common.parseYear(bib.publicationDate)
        number = bib.issue
        issn = bib.issn
    }

    companion object {
        fun getFullJournalInfo(journal: JournalRef) {
            val bibitem = GrobidEngine.processRawReference(journal.rawRef, 0)
            journal.rawTitle = bibitem.journal
            val (fp, lp) = Common.splitPages(bibitem.pageRange)
            journal.firstPage = fp
            journal.lastPage = lp
            journal.volume = bibitem.volumeBlock
            journal.year = Common.parseYear(bibitem.publicationDate)
            journal.number = bibitem.issue
            journal.issn = bibitem.issn
        }
    }
}