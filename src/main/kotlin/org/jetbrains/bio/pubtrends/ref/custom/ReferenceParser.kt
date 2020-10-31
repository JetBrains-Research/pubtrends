package org.jetbrains.bio.pubtrends.ref.custom

import org.jetbrains.bio.pubtrends.ref.GrobidEngine
import org.jetbrains.bio.pubtrends.data.Reference
import org.jetbrains.bio.pubtrends.validation.ArxivValidator
import org.apache.logging.log4j.kotlin.logger

object ReferenceParser {
    private val logger = logger()
    fun parse(
        lines: List<Line>,
        refType: ReferenceType,
        isTwoColumns: Boolean,
        pageWidth: Int
    ): List<String> {
        logger.debug("Begin reference parsing")
        val refList = mutableListOf<String>()
        val refRegex = refType.regex
        if (!isTwoColumns) {
            //analyze this text
            var canUseSecondIndentPattern = true
            var i = 0
            var secondLineIndent = -1
            var curRefNum = 1
            val firstLineIndices = mutableListOf<Int>()
            var maxWidth = 0
            while (i < lines.size) {
                firstLineIndices.add(i)
                //find next reference
                maxWidth = Integer.max(maxWidth, lines[i].lastPos - lines[i].indent)
                var j = i + 1
                while (j < lines.size) {
                    val match = refRegex.find(lines[j].str)
                    if (match != null) {
                        if (refType.isNum) {
                            val value = match.value.drop(refType.firstLen).dropLast(refType.lastLen).toInt()
                            if (value == curRefNum + 1) {
                                if (refType.strict && lines[j].indent == secondLineIndent) {
                                    logger.debug("Drop because first line indent is equal to second line indent")
                                    return listOf()
                                }
                                else {
                                    break
                                }
                            }
                            else if (value == curRefNum || lines[j].indent != secondLineIndent && !refType.strict) {
                                logger.debug("Drop because of error in reference numeration")
                                return listOf()
                            }
                        }
                        else if (refType.strict && lines[j].indent == secondLineIndent) {
                            logger.debug("Drop because first line indent is equal to second line indent")
                            return listOf()
                        } else {
                            break
                        }
                    }
                    j += 1
                }

                if (j != lines.size) {
                    for (k in i + 1 until j) {
                        if (secondLineIndent == -1) {
                            secondLineIndent = lines[k].indent
                        }
                        if (secondLineIndent != lines[k].indent) {
                            canUseSecondIndentPattern = false
                        }
                    }
                } else {
                    //this was the last reference
                    break
                }
                curRefNum += 1
                i = j
            }

            if (secondLineIndent == -1) {
                canUseSecondIndentPattern = false
            }

            if (refType.strict && !canUseSecondIndentPattern) {
                logger.debug("Drop because can't use indent pattern")
                return listOf()
            }

            logger.debug("Found $curRefNum references")

            //parse references
            for ((j, lineInd) in firstLineIndices.withIndex()) {
                var curRef = ""
                var containMultipleReferences = false
                if (j != firstLineIndices.lastIndex) {
                    val nextLineInd = firstLineIndices[j + 1]
                    for (k in lineInd until nextLineInd) {
                        val newLine = if (k == lineInd) removeRefPattern(lines[k].str, refRegex) else lines[k].str
                        if (newLine == null) {
                            return listOf()
                        }
                        curRef = addLineToReference(curRef, newLine)

                        if (curRef.isNotBlank() && curRef.last() != ';'
                            && (curRef.last() == '.' || nextLineInd - lineInd > 5 || containMultipleReferences)
                            && (curRef.last() != ',' && !ArxivValidator.containsId(newLine))
                            && ((k != lineInd && lines[k].lastPos < lines[k - 1].lastPos * 0.9)
                                    || k == lineInd && (lines[k].lastPos - lines[k].indent) < maxWidth * 0.9)
                        ) {
                            //small checking if some of the lines we want to throw away contains arxivId or year
                            //and then add semicolon to identify them as separate references later
                            if (lines.subList(lineInd + 1, nextLineInd).any {
                                    ArxivValidator.containsId(it.str) || it.str.contains("""(19|20)\d\d""".toRegex())
                                }) {
                                curRef += ';'
                                containMultipleReferences = true
                            } else {
                                //then this is the end of reference
                                break
                            }
                        }
                    }
                } else {
                    var linesParsed = 0
                    //this is the last reference and we should find it's end
                    for (k in lineInd until lines.size) {
                        val newLine = if (k == lineInd) removeRefPattern(lines[k].str, refRegex) else lines[k].str
                        if (newLine == null) {
                            return listOf()
                        }
                        linesParsed += 1
                        if (linesParsed > 5) {
                            return listOf()
                        }
                        curRef = addLineToReference(curRef, newLine)
                        if (canUseSecondIndentPattern && k < lines.lastIndex &&
                            lines[k + 1].indent != secondLineIndent
                        ) {
                            //we find the end of reference
                            break
                        }
                        if (k > lineInd) {
                            if (lines[k].lastPos < lines[k - 1].lastPos * 0.9) {
                                //we find the end of reference
                                break
                            }
                        } else {
                            if (lines[k].lastPos - lines[k].indent < 0.9 * maxWidth
                                && (k + 1 >= lines.size || !ArxivValidator.containsId(lines[k + 1].str))
                                && curRef.last() != ',') {
                                //we find the end of reference
                                break
                            }
                        }
                    }
                }
                val refs = curRef.split(";").map{it.trim()}.filter { it.isNotEmpty() }
                if (refs.size > 1) {
                    if (refs.any { rejectAsReferenceStrong(it) }) {
                        logger.debug("Drop because can't parse reference with semicolon")
                        return listOf()
                    }
                }
                if (refs.any { rejectAsReferenceWeak(it) }) {
                    logger.debug("Drop because multiple references was parsed as one")
                    return listOf()
                }
                refList.addAll(refs)
            }
        } else {
            var canUseSecondIndentPattern = true
            var i = 0
            var secondLineIndentLeft = -1
            var secondLineIndentRight = -1

            //current page side(0 -- left, 1 -- right)
            fun getSide(line: Line): Int {
                return if (line.indent < pageWidth * 0.4) 0 else 1
            }

            var curRefNum = 1
            val firstLineIndices = mutableListOf<Int>()
            var maxWidth = 0
            //analyze this text
            while (i < lines.size) {
                firstLineIndices.add(i)
                //find next reference
                maxWidth = Integer.max(maxWidth, lines[i].lastPos - lines[i].indent)
                var j = i + 1
                while (j < lines.size) {
                    val match = refRegex.find(lines[j].str)
                    val secondLineIndent = when {
                        getSide(lines[j]) == 0 -> secondLineIndentLeft
                        else                   -> secondLineIndentRight
                    }
                    if (match != null) {
                        if (refType.isNum) {
                            val value = match.value.drop(refType.firstLen).dropLast(refType.lastLen).toInt()
                            if (value == curRefNum + 1) {
                                if (!refType.strict || lines[j].indent != secondLineIndent) {
                                    break
                                }
                                else {
                                    logger.debug("Drop because first line indent is equal to second line indent")
                                    return listOf()
                                }
                            }
                            else if (value == curRefNum || lines[j].indent != secondLineIndent && !refType.strict) {
                                logger.debug("Drop because of error in reference numeration")
                                return listOf()
                            }
                        }
                        else if (refType.strict && lines[j].indent == secondLineIndent) {
                            logger.debug("Drop because first line indent is equal to second line indent")
                            return listOf()
                        } else {
                            break
                        }
                    }
                    j += 1
                }
                if (j != lines.size) {
                    for (k in i + 1 until j) {
                        val curSide = getSide(lines[k])
                        if (curSide == 0) {
                            if (secondLineIndentLeft == -1) {
                                secondLineIndentLeft = lines[k].indent
                            }
                            if (secondLineIndentLeft != lines[k].indent) {
                                canUseSecondIndentPattern = false
                            }
                        } else {
                            if (secondLineIndentRight == -1) {
                                secondLineIndentRight = lines[k].indent
                            }
                            if (secondLineIndentRight != lines[k].indent) {
                                canUseSecondIndentPattern = false
                            }
                        }
                    }
                } else {
                    //this was the last reference
                    break
                }
                curRefNum += 1
                i = j
            }

            if (secondLineIndentLeft == -1 && secondLineIndentRight == -1) {
                canUseSecondIndentPattern = false
            }

            if (refType.strict && !canUseSecondIndentPattern) {
                logger.debug("Drop because can't use indent pattern")
                return listOf()
            }

            logger.debug("Found $curRefNum reference lines")

            //parse references
            for ((j, lineInd) in firstLineIndices.withIndex()) {
                var curRef = ""
                if (j != firstLineIndices.lastIndex) {
                    val nextLineInd = firstLineIndices[j + 1]
                    var prevSide = 0
                    var containsMultipleReferences = false
                    for (k in lineInd until nextLineInd) {
                        val newLine = if (k == lineInd) removeRefPattern(lines[k].str, refRegex) else lines[k].str
                        if (newLine == null) {
                            return listOf()
                        }
                        curRef = addLineToReference(curRef, newLine)
                        val curSide = getSide(lines[k])
                        if (curRef.isNotBlank() && curRef.last() != ';'
                            && (curRef.last() == '.' || nextLineInd - lineInd > 10 || containsMultipleReferences)
                            && ((k != lineInd && curSide == prevSide && lines[k].lastPos < lines[k - 1].lastPos * 0.9)
                                    || (lines[k].lastPos - lines[k].indent) < maxWidth * 0.8)
                        ) {

                            if (lines.subList(lineInd + 1, nextLineInd).any {
                                    ArxivValidator.containsId(it.str) || it.str.contains("""(19|20)\d\d""".toRegex())
                                }) {
                                curRef += ';'
                                containsMultipleReferences = true
                            } else {
                                //then this is the end of reference
                                break
                            }
                        }
                        prevSide = curSide
                    }
                } else {
                    //this is the last reference and we should find it's end
                    var prevSide = 0
                    var linesParsed = 0
                    for (k in lineInd until lines.size) {
                        val newLine = if (k == lineInd) removeRefPattern(lines[k].str, refRegex) else lines[k].str
                        if (newLine == null) {
                            return listOf()
                        }
                        linesParsed += 1
                        if (linesParsed > 9) {
                            return listOf()
                        }
                        curRef = addLineToReference(curRef, newLine)
                        val curSide = getSide(lines[k])
                        if (canUseSecondIndentPattern && k < lines.lastIndex) {
                            if (lines[k + 1].indent != secondLineIndentLeft
                                && lines[k + 1].indent != secondLineIndentRight
                            ) {

                                //we find the end of reference
                                break
                            }
                        }
                        if (k > lineInd) {
                            if (prevSide == curSide && lines[k].lastPos < lines[k - 1].lastPos * 0.9) {
                                //we find the end of reference
                                break
                            }
                        } else {
                            if (lines[k].lastPos - lines[k].indent < 0.8 * maxWidth
                                && (k + 1 >= lines.size || !ArxivValidator.containsId(lines[k + 1].str))
                                && curRef.last() != ','
                            ) {
                                //we find the end of reference
                                break
                            }
                        }
                        prevSide = curSide
                    }
                }
                val refs = curRef.split(";").map{it.trim()}.filter { it.isNotEmpty() }
                if (refs.size > 1) {
                    if (refs.any { rejectAsReferenceStrong(it) }) {
                        logger.debug("Drop because can't parse reference with semicolon")
                        return listOf()
                    }
                }
                if (refs.any { rejectAsReferenceWeak(it) }) {
                    logger.debug("Drop because multiple references was parsed as one")
                    return listOf()
                }
                refList.addAll(refs)
            }
        }
        return refList
    }

    private fun addLineToReference(ref: String, line: String): String {
        return if (ref.length > 2 && ref.last() == '-') {
            val wordPart1 = ref.takeLastWhile { !it.isWhitespace() }.drop(1).all {it.isLowerCase()}
            val wordPart2 = line.takeWhile { !it.isWhitespace() }.all{it.isLowerCase()}
            if (wordPart1 && wordPart2) {
                ref.dropLast(1) + line
            } else {
                ref + line
            }
        } else {
            if (ref == "") {
                line
            } else {
                ref + " " + line
            }
        }
    }

    private fun removeRefPattern(ref : String, regex : Regex) : String? {
        val match = regex.find(ref) ?: return null
        return ref.drop(match.range.last + 1).trimIndent()
    }

    private fun rejectAsReferenceStrong(ref : String) : Boolean {
        val reference =
            Reference(ref, GrobidEngine.processRawReference(ref, 0))
        var containsAuthors = false
        var containsJournal = false
        var containsYear = false
        var containsArxivId = false
        if (!reference.authors.isNullOrEmpty()) {
            containsAuthors = true
        }
        if (!reference.journal.isNullOrEmpty()) {
            containsJournal = true
        }
        if (reference.year != null) {
            containsYear = true
        }
        if (ArxivValidator.containsId(ref)) {
            containsArxivId = true
        }
        return !(containsJournal ||
                containsYear ||
                containsArxivId && containsAuthors)
    }

    private fun rejectAsReferenceWeak(ref: String): Boolean {
        return ArxivValidator.containsMultipleIds(ref)
                || """\((19|20)\d\d\)""".toRegex().findAll(ref).toList().size > 1
    }
}