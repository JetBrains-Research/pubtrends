package com.preprint.server.core.ref.custom

enum class ReferenceType(val regex : Regex,
                         val firstRegex : Regex,
                         val num : Int,
                         val firstLen : Int,
                         val lastLen : Int,
                         val isNum : Boolean,
                         val strict : Boolean)
{
    A("""^\[\d{1,4}]""".toRegex(),
        """^\[1]""".toRegex(),
        0, 1, 1, true, false),
    B("""^\d{1,4}\.""".toRegex(),
        """1\.""".toRegex(),
        1, 0, 1, true, true),
    C("""^\d{1,4}""".toRegex(),
        """1 """.toRegex(),
        2, 0, 0, true, true),
    D("""^\[.*?]""".toRegex(),
        """^\[.*?]""".toRegex(),
        3, 1, 1, false, true)
}