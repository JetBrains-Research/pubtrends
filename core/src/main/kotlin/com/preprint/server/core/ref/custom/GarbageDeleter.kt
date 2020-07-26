package com.preprint.server.core.ref.custom

import com.preprint.server.core.algo.Algorithms
import java.lang.Math.abs


object GarbageDeleter {
    fun removePageNumbers(lines : List<Line>) : List<Line> {
        //find out where page numbers are located(bottom or top or alternate)

        val pageNumberPos = mutableListOf<Int>()
        for ((i, line) in lines.withIndex()) {
            if (line.pn == 1) {
                continue
            }
            if (line.indent == PdfMarks.PageStart.num) {
                if (i + 1 <= lines.lastIndex && lines[i + 1].str.contains(line.pn.toString())) {
                    pageNumberPos.add(0)
                }
            }
            if (i + 1 <= lines.lastIndex && lines[i + 1].indent == PdfMarks.PageEnd.num) {
                if (line.str.contains(line.pn.toString())) {
                    if (pageNumberPos.size + 1 == line.pn) {
                        //the this page has page number in the first and in the last line
                        //mark this page with 2
                        pageNumberPos[pageNumberPos.lastIndex] = 2
                    }
                    else {
                        //this page has page number in the last line
                        pageNumberPos.add(1)
                    }
                }
                if (pageNumberPos.size + 1 < line.pn) {
                    //then this page doesn't contain page number
                    //and we will assume that all pages doesn't contain page number
                    return removePageNumbersSimple(lines)
                }
            }

            //we only want to scan first 6 pages
            if (line.pn == 8) {
                break
            }
        }
        //0 -- page number in the first line
        //1 -- page number in the second line
        //2 -- in the odd pages in the first line, in the even on the last
        //3 -- in the even pages in the last line, in the odd on the first
        //4 -- in the first and in the last line
        val pagePattern : Int
        if (pageNumberPos.size < 6) {
            when {
                (pageNumberPos.all {it == 0}) -> pagePattern = 0
                (pageNumberPos.all {it == 1}) -> pagePattern = 1
                else                          -> pagePattern = -1 //we haven't got enough information
            }
        }
        else {
            when {
                (pageNumberPos.all {it == 2}) -> pagePattern = 4

                (pageNumberPos.all {it == 0 || it == 2})      -> pagePattern = 0

                (pageNumberPos.all {it == 1 || it == 2})      -> pagePattern = 1

                (pageNumberPos.withIndex().all
                {(i, p) -> (i % 2 == 0 && (p == 0 || p == 2))
                        || i % 2 == 1 && (p == 1 || p == 2)}) -> pagePattern = 2

                (pageNumberPos.withIndex().all
                {(i, p) -> (i % 2 == 0 && (p == 1 || p == 2))
                        || i % 2 == 1 && (p == 0 || p == 2)}) -> pagePattern = 3

                else                                          -> pagePattern = -1
            }
        }

        if (pagePattern == -1) {
            return removePageNumbersSimple(lines)
        }

        //(is first or last line, line, page number) -> should we delete this line or not
        val deleter : (Boolean, Line) -> Boolean = when(pagePattern) {
            0 -> {
                    isFirst, line -> isFirst && (line.str.contains(line.pn.toString()))
            }
            1 -> {
                    isFirst, line -> !isFirst && (line.str.contains(line.pn.toString()))
            }
            2 -> {
                    isFirst, line -> isFirst && line.pn % 2 == 1 || !isFirst && line.pn % 2 == 0
            }
            3 -> {
                    isFirst, line -> isFirst && line.pn % 2 == 0 || !isFirst && line.pn % 1 == 0
            }
            else -> {
                    _, _ -> true
            }
        }

        return lines.filterIndexed{i, line ->
            if (i - 1 > 0 && lines[i - 1].indent == PdfMarks.PageStart.num) {
                !deleter(true, line)
            }
            else if (i + 1 <= lines.lastIndex && lines[i + 1].indent == PdfMarks.PageEnd.num) {
                !deleter(false, line)
            } else {
                true
            }
        }
    }

    fun removePageNumbersSimple(lines : List<Line>) : List<Line> {
        val firstLineIndices = getFirstLineIndices(lines).drop(1)
        val lastLineIndices = getLastLineIndices(lines).drop(1)

        //drop first page
        firstLineIndices.drop(1)
        lastLineIndices.drop(1)

        fun isDigits(indices : List<Int>) : Boolean {
            return indices.all{ind ->
                    lines[ind].str.all {it.isDigit()}
            }
        }

        return when {
            isDigits(firstLineIndices) -> lines.filterIndexed {i, line ->
                !(i - 1 >= 0 && lines[i - 1].indent == PdfMarks.PageStart.num)
            }
            isDigits(lastLineIndices) -> lines.filterIndexed {i, line ->
                !(i + 1 <= lines.lastIndex && lines[i + 1].indent == PdfMarks.PageEnd.num)
            }
            else -> lines
        }
    }

    fun removePageHeaders(lines : List<Line>) : List<Line> {
        //find longest common substring


        //make all lines with indices from the list, that contains headers, empty
        fun removeHeaders(listIndices : List<Int>) {
            //capture 'lines' from outer function

            var state = 0
            //current longest substring for the last lines
            var curMaxString = ""
            var runLength = 0
            for (i in 1 until listIndices.size) {
                val cur = listIndices[i]
                val prev = listIndices[i - 1]
                if (state == 0) {
                    curMaxString = Algorithms.findLCS(lines[prev].str, lines[cur].str)
                    if (curMaxString.length > lines[cur].str.length * 0.75 &&
                        curMaxString.length > lines[prev].str.length * 0.75 && curMaxString.length > 1
                    ) {
                        //then we assume that this strings contain header
                        state = 1
                        runLength = 2
                    } else {
                        curMaxString = ""
                    }
                    continue
                }
                if (state == 1) {
                    val newString = Algorithms.findLCS(curMaxString, lines[cur].str)
                    if (newString.length > lines[cur].str.length * 0.75
                        && (newString.length == curMaxString.length
                        || curMaxString.length > 15 && abs(curMaxString.length - newString.length) < 4)
                    ) {
                        runLength += 1
                        if (i == listIndices.lastIndex) {
                            if (runLength >= 3) {
                                for (j in i downTo (i - runLength + 1)) {
                                    lines[listIndices[j]].str = ""
                                }
                            }
                        }
                        continue
                    }
                    if (runLength >= 3) {
                        for (j in i - 1 downTo (i - runLength)) {
                            lines[listIndices[j]].str = ""
                        }
                    }
                    state = 0
                    runLength = 0
                }
            }
        }

        val firstLineInd = getFirstLineIndices(lines)
        val lastLineInd = getLastLineIndices(lines)
        val evenFirstLineInd = firstLineInd.filterIndexed{i, _ -> i % 2 == 0}
        val oddFirstLineInd = firstLineInd.filterIndexed {i, _ -> i % 2 == 1}
        val evenLastLineInd = lastLineInd.filterIndexed{i, _ -> i % 2 == 0}
        val oddLastLineInd = lastLineInd.filterIndexed {i, _ -> i % 2 == 1}
        removeHeaders(firstLineInd)
        removeHeaders(lastLineInd)
        removeHeaders(evenFirstLineInd)
        removeHeaders(evenLastLineInd)
        removeHeaders(oddFirstLineInd)
        removeHeaders(oddLastLineInd)
        return lines.filter { it.str != "" }
    }

    private fun getFirstLineIndices(lines : List<Line>) : List<Int> {
        return lines.mapIndexed { i, line ->
            if (line.indent == PdfMarks.PageStart.num && i + 1 < lines.size) {
                i + 1
            }
            else {
                -1
            }
        }.filter{ it != -1 }
    }

    private fun getLastLineIndices(lines : List<Line>) : List<Int> {
        return lines.mapIndexed { i, line ->
            if (line.indent == PdfMarks.PageEnd.num && i - 1 >= 0) {
                i - 1
            }
            else {
                -1
            }
        }.filter{ it != -1 }
    }
}