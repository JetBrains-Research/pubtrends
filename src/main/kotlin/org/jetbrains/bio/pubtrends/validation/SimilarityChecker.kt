package org.jetbrains.bio.pubtrends.validation

import org.jetbrains.bio.pubtrends.algo.Algorithms
import org.jetbrains.bio.pubtrends.data.Reference

object SimilarityChecker {
    fun check(ref: Reference, record: ValidationRecord): Boolean {
        val refstr = ref.rawReference
        var score = 4
        if (ref.year != null && record.year != null && Math.abs(record.year - ref.year!!) > 1) {
            return false
        }

        if (record.year != null
            && record.year != ref.year
        ) {
            score -= 1
        }

        if (record.journalVolume != null
            && record.journalVolume != ref.volume
        ) {
            score -= 1
        }

        if (record.firstPage != null
            && record.firstPage != ref.firstPage
        ) {
            score -= 1
        }

        //checking authors
        if (!record.authors.all {author -> containsAuthor(author.name, refstr) }) {
            score -= 2
        }

        if (score < 2) {
            return false
        }

        if (record.lastPage != null) {
            if (record.lastPage == ref.lastPage) {
                score += 1
            }
        }

        if (record.journalName != null) {
            val t = Algorithms.findLCS(refstr, record.journalName).length
            if (t.toDouble() / record.journalName.length.toDouble() > 0.8) {
                score += 1
            }
        }

        if (record.issue != null) {
            if (record.issue == ref.issue) {
                score += 1
            }
        }

        if (!record.title.isNullOrBlank()) {
            val t = Algorithms.findLCS(refstr, record.title).length
            if (t.toDouble() / record.title.length.toDouble() > 0.9) {
                score += 2
            }
        }
        if (refstr.contains(record.firstPage.toString())) {
            score += 1
        }
        return score >= 4
    }

    fun checkByTitle(ref: Reference, record: ValidationRecord): Boolean {
        if (
            record.authors.all { containsAuthor(it.name, ref.rawReference) }
            && ref.rawReference.contains(record.year.toString())
        ) {
            return true
        }

        if (ref.title?.trim() == record.title?.trim()) {
            return true
        }

        return false
    }

    private fun containsAuthor(author: String, refstr: String): Boolean {
        val authorParts = author.split("""\s""".toRegex()).filter { it.isNotBlank() }
        val longestPart = authorParts.maxBy { it.length }
        if (longestPart != null) {
            return Algorithms.findLCS(refstr, longestPart).length + 1 >= longestPart.length
                    || (authorParts.last().all {it.isLetter()} && authorParts.last().length > 1 &&
                    Algorithms.findLCS(refstr, authorParts.last()).length + 1 >= authorParts.last().length)
        }
        else {
            return false
        }
    }
}