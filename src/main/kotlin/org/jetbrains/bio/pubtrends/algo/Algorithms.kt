package com.preprint.server.core.algo


/**
 * Contains a set of simple algorithms used in the project
 */
object Algorithms {

    /**
     * Finds Levenshtein distance between two strings
     */
    fun findLvnstnDist(str1: String, str2: String, ignorCase: Boolean = true) : Int {
        val n = str1.length
        val m = str2.length
        val dp = Array(n + 1, {IntArray(m + 1, {0})})
        for (i in 1 until n + 1) {
            for (j in 1 until m + 1) {
                dp[i][j] = Integer.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
                val a = if (ignorCase) str1[i - 1].toLowerCase() else str1[i - 1]
                val b = if (ignorCase) str2[j - 1].toLowerCase() else str2[j - 1]
                if (a == b) {
                    dp[i][j] = Integer.min(dp[i][j], dp[i - 1][j - 1])
                }
                else {
                    dp[i][j] = Integer.min(dp[i][j], dp[i - 1][j - 1] + 1)
                }
            }
        }
        return dp[n][m]
    }


    /**
     * Finds longest common substring of two strings
     */
    fun findLCS(s1 : String, s2 : String) : String {
        if (s1 == s2) {
            return s1
        }
        val dp = Array(s1.length + 1) {IntArray(s2.length + 1)}
        for (i in 0 until s1.length + 1) {
            for (j in 0 until s2.length + 1) {
                dp[i][j] = 0
            }
        }
        for (i in 1 until s1.length + 1) {
            for (j in 1 until s2.length + 1) {
                dp[i][j] = Integer.max(dp[i - 1][j], dp[i][j - 1])
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = Integer.max(dp[i][j], dp[i - 1][j - 1] + 1)
                }
            }
        }
        var li : Int = s1.length
        var lj : Int = s2.length
        var res = ""
        while (li > 0 && lj > 0) {
            if (dp[li][lj] == dp[li - 1][lj]) {
                li -= 1
                continue
            }
            if (dp[li][lj] == dp[li][lj - 1]) {
                lj -= 1
                continue
            }
            res += s1[li - 1]
            li -= 1
            lj -= 1
        }
        return res.reversed()
    }
}