package com.preprint.server.core

import com.preprint.server.core.data.Reference
import com.preprint.server.core.validation.ArxivValidator
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ArxivValidatorTests {
    @Test
    fun testAll() {
        val refs = listOf(
            Reference("lalalalalalalalalal arXiv:cond-mat/9701102 lalalala"),
            Reference("lalalalalalalalalal cond-mat/9701102 lalalala"),
            Reference("lalalalalalalalalal arXiv:cond-mat/9701102v1 lalalala"),
            Reference("lalalalalalalalalal cond-mat/9701102v2 lalalala"),
            Reference("lalalalalalalalalal arXiv:2001.00061 lalalalal"),
            Reference("lalalalalalalalalal arXiv:2001.00061v2 lalalalal")
        )
        val expected = listOf(
            "cond-mat/9701102",
            "cond-mat/9701102",
            "cond-mat/9701102",
            "cond-mat/9701102",
            "2001.00061",
            "2001.00061"
        )
        runBlocking { ArxivValidator.validate(refs) }

        assertEquals(expected, refs.map {it.arxivId})
    }
}