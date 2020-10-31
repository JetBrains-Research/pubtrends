package com.preprint.server.core.validation

data class ValidationRecord(
    val authors: List<Author> = mutableListOf(),
    val journalName: String? = null,
    val journalVolume: String? = null,
    val title: String? = null,
    val year: Int? = null,
    val firstPage: Int? = null,
    val lastPage: Int? = null,
    val issue: String? = null
) {
    data class Author(
        val name: String = ""
    )
}