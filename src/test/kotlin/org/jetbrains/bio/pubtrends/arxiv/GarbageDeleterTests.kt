package com.preprint.server.core

import com.preprint.server.core.ref.CustomReferenceExtractor
import com.preprint.server.core.ref.custom.GarbageDeleter
import io.mockk.*
import org.apache.logging.log4j.kotlin.KotlinLogger
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.assertEquals

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class GarbageDeleterTests {

    @BeforeAll
    fun initTest() {
        mockkConstructor(KotlinLogger::class)
        every {anyConstructed<KotlinLogger>().info(any<String>())} just Runs
    }

    @AfterAll
    fun clear() {
        unmockkAll()
    }

    fun loadFromResourses(fileName : String) : String {
        return this.javaClass.getResource("/garbageDeleterTests/$fileName").readText()
    }

    @Test
    fun removePageNumbersLowerTest() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum1"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum1.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }

    @Test
    fun removePageNumbersUpperTest() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum2"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum2.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }

    @Test
    fun removePageNumbersMixedTest() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum3"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum3.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }

    @Test
    fun removePageNumbersFewPagesTest() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum4"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageNum4.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }

    @Test
    fun removePageHeadersTestUpper() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageHead1"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageHead1.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }

    @Test
    fun removePageHeadersTestLower() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageHead2"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageHead2.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }

    @Test
    fun removePageHeadersTestMixed() {
        val textLines = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageHead3"))
        val expected = CustomReferenceExtractor.getLines(loadFromResourses("garbagePageHead3.exp"))
        val actual = GarbageDeleter.removePageNumbers(textLines)

        assertEquals(expected, actual)
    }
}