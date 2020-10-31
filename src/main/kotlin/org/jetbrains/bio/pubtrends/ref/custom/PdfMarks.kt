package com.preprint.server.core.ref.custom

enum class PdfMarks(val str : String, val num: Int) {
    PageStart("\\@ps\\", -1),
    PageEnd("\\@pe\\", -2),
    RareFont("\\r\\", 0),
    IntBeg("@d", 0),
    IntEnd("@d", 0),
}