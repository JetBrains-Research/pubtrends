package com.preprint.server.core.validation

import com.preprint.server.core.data.Reference


/**
 * Parses arxiv if from the reference string
 */
object ArxivValidator : Validator {
    val ids = this.javaClass.getResource("/ids.txt").readText().lines()


    /**
     * Searches for arxiv id in the string
     * and stores it in the `arxivId` property of `ref`.
     * Also deletes author and journal properties if arxiv id was found
     * (read description of `resetAllData` method)
     */
    override fun validate(ref: Reference) {
        ref.arxivId = ""
        val beg = ref.rawReference.lastIndexOf("arxiv:", ignoreCase = true)
        if (beg != -1 && beg + 6 < ref.rawReference.length && ref.rawReference[beg + 6].isDigit()) {
            val arx = ref.rawReference.substring(beg).substringAfter(":")
            if (arx.length > 8) {
                var res : String? = null
                if (arx.substring(0, 4).all {it.isDigit()} && arx[4] == '.') {
                    res = arx.substring(0, 5)
                    for (c in arx.substring(5)) {
                        if (!c.isDigit()) break
                        res += c
                    }
                }
                else {
                    val i = arx.indexOf('/')
                    if (i != -1) {
                        res = arx.substring(0, i + 1)
                        for (c in arx.substring(i + 1)) {
                            if (!c.isDigit()) {
                                break
                            }
                            res += c
                        }
                    }
                }

                if (res != null) {
                    ref.arxivId = res
                    resetAllData(ref)
                }
            }
        }
        else {
            ids.forEach {idPrefix ->
                val beg = ref.rawReference.indexOf(idPrefix + "/")
                if (beg != -1) {
                    var res = idPrefix + "/"
                    var i = ref.rawReference.indexOf('/', beg) + 1
                    if (i < ref.rawReference.length && ref.rawReference[i].isDigit()) {
                        while (i < ref.rawReference.length && ref.rawReference[i].isDigit()) {
                            res += ref.rawReference[i]
                            i += 1
                        }
                        if (res.isNotBlank()) {
                            ref.arxivId = res
                            resetAllData(ref)
                        }
                    }
                    return
                }
                else {
                    val beg = """$idPrefix\.\p{Upper}{2}/""".toRegex().find(ref.rawReference)
                    if (beg != null) {
                        var i = beg.range.last + 1
                        if (i < ref.rawReference.length && ref.rawReference[i].isDigit()) {
                            var res = idPrefix + "/"
                            while (i < ref.rawReference.length && ref.rawReference[i].isDigit()) {
                                res += ref.rawReference[i]
                                i += 1
                            }
                            if (res.isNotBlank()) {
                                ref.arxivId = res
                                resetAllData(ref)
                            }
                        }
                        return
                    }
                }
            }
        }
    }

    /** When this record(reference) will be stored in the database
     * we don't want to create author and journal connections,
     * because when record with this arxiv id will be loaded
     * from arxiv.org or Amazon S3 we will create this connections anyway.
     * And if we don't delete them now, duplicates may occur
     */
    private fun resetAllData(ref: Reference) {
        if (!ref.validated) {
            ref.validated = true
            ref.isReference = true
            ref.authors = mutableListOf()
            ref.journal = null
        }
    }

    fun containsId(ref : String) : Boolean {
        val lref = ref.toLowerCase()
        return ids.any {lref.contains("""$it(\.\p{Upper}{2})?/\d\d\d""".toRegex()) || lref.contains("arxiv:")}
    }

    fun containsMultipleIds(ref: String): Boolean {
        val lref = ref.toLowerCase()
        return ids.sumBy { """$it(\.\p{Upper}{2})?/\d\d\d""".toRegex().findAll(lref).toList().size } > 1 ||
                "arxiv:".toRegex().findAll(lref).toList().size > 1
    }
}