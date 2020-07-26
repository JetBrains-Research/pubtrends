package com.preprint.server.core.utils

object Common {
    fun splitPages(pages: String?): Pair<Int?, Int?> {
        if (pages == null) {
            return Pair(null, null)
        }
        var i = pages.indexOfFirst { it.isDigit() }
        if (i == -1) {
            return Pair(null, null)
        }
        var firstPageString = ""
        var lastPageString = ""
        while (i < pages.length && pages[i].isDigit()) {
            firstPageString += pages[i]
            i += 1
        }

        while (i < pages.length && !pages[i].isDigit()) {
            i += 1
        }

        while (i < pages.length && pages[i].isDigit()) {
            lastPageString += pages[i]
            i += 1
        }
        return Pair(firstPageString.toIntOrNull(), lastPageString.toIntOrNull())
    }

    fun parseYear(dataStr: String?): Int? {
        if (!dataStr.isNullOrBlank()) {
            val yearRegex = """(19|20)\d\d""".toRegex()
            val match = yearRegex.find(dataStr)
            if (match != null) {
                return match.value.toIntOrNull()
            }
        }
        return null
    }
}