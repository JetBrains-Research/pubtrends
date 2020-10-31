package org.jetbrains.bio.pubtrends.ref.custom

import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.text.TextPosition
import java.lang.Math.round


object PDFRefTextStripper: PDFTextStripper() {
    val fontWidthToCnt = mutableMapOf<Int, Int>()
    var lastY = 0f
    var lastPageNo = 0
    var isTwoColumns = false

    init {
        //set mark for the end of the page
        super.setPageEnd("\n${PdfMarks.PageEnd.str}\n")
        //set mark for the start of the page
        super.setPageStart("\n${PdfMarks.PageStart.str}\n")

    }

    override fun writeString(text: String?, textPositions: MutableList<TextPosition>?) {
        if (text == null || textPositions == null || textPositions.size == 0) {
            return
        }
        var newText = text
        val curY = textPositions[0].y
        val curX = textPositions[0].x
        val curPageNo = currentPageNo
        val pageHeight = currentPage.bBox.height
        val pageWidth = currentPage.bBox.width
        val curFontWidth = round(textPositions.last().font.boundingBox.width)
        for ((i, symbol) in textPositions.withIndex()) {
            val curFontWidth = round(symbol.font.boundingBox.width)
            fontWidthToCnt[curFontWidth] = (fontWidthToCnt[curFontWidth] ?: 0) + 1
        }

        //check if this document has two columns
        if (!isTwoColumns && curPageNo == lastPageNo && curY < lastY
            && lastY - curY > pageWidth / 2 && curX > pageWidth / 2.3) {
            isTwoColumns = true
        }

        //check if the font differs from standard document font and find word 'References'
        if (text.all {it.isUpperCase()} || curFontWidth != fontWidthToCnt.maxBy{it.value}!!.key ||
                textPositions[0].font.fontDescriptor?.fontName?.contains("bold",ignoreCase = true) ?: false) {
            val pos1 = text.indexOf("References")
            val pos2 = text.indexOf("REFERENCES")
            val pos = if (pos1 != -1) pos1 else pos2
            if (pos != -1) {
                //mark bold word 'References'
                newText = newText.substring(0, pos) + PdfMarks.RareFont.str + newText.substring(pos)
            }
        }

        //write the coordinate of the word
        newText = PdfMarks.IntBeg.str + round(textPositions[0].x).toString() + PdfMarks.IntEnd.str + newText

        //write the last coordinte of the word
        newText = newText + PdfMarks.IntBeg.str + round(textPositions.last().x).toString() + PdfMarks.IntEnd.str

        lastPageNo = curPageNo
        lastY = curY
        super.writeString(newText, textPositions)
    }

    fun getMarkedText(doc: PDDocument?): String {
        isTwoColumns = false
        lastPageNo = 0
        lastY = 0f
        fontWidthToCnt.clear()
        return super.getText(doc)
    }
}
