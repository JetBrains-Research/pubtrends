package com.preprint.server.core.crossref

import com.beust.klaxon.Json

/**
 * This class and all subclasses are auto-generated
 * and used to parse data from CrossRef API response
 */
data class CrossRefJsonData(
    val message: Message? = null,
    @Json(name="message-type")
    val message_type: String? = null,
    @Json(name="message-version")
    val message_version: String? = null,
    val status: String? = null
) {
    data class Message(
        val facets: Facets? = null,
        val items: List<Item>? = null,
        @Json(name="items-per-page")
        val items_per_page: Int? = null,
        val query: Query? = null,
        val total_results: Int? = null
    )

    class Facets(
    )

    data class Item(
        val DOI: String? = null,
        val ISSN: List<String>? = null,
        val URL: String? = null,
        @Json(name="alternative-id")
        val alternative_id: List<String>? = null,
        val author: List<Auth>? = null,
        @Json(name="container-title")
        val container_title: List<String>? = null,
        @Json(name="content-domain")
        val content_domain: ContentDomain? = null,
        val created: Created? = null,
        val deposited: Deposited? = null,
        val indexed: Indexed? = null,
        @Json(name="is-referenced-by-count")
        val is_referenced_by_count: Int? = null,
        @Json(name="issn-type")
        val issn_type: List<IssnType>? = null,
        val issue: String? = null,
        val issued: Issued? = null,
        @Json(name="journal-issue")
        val journal_issue: JournalIssue? = null,
        val language: String? = null,
        val license: List<License>? = null,
        val link: List<Link>? = null,
        val member: String? = null,
        val page: String? = null,
        val prefix: String? = null,
        @Json(name="published-online")
        val published_online: PublishedOnline? = null,
        @Json(name="published-print")
        val published_print: PublishedPrintX? = null,
        val publisher: String? = null,
        val reference: List<Ref>? = null,
        @Json(name="reference-count")
        val reference_count: Int? = null,
        @Json(name="references-count")
        val references_count: Int? = null,
        val relation: Relation? = null,
        val score: Double? = null,
        @Json(name="short-container-title")
        val short_container_title: List<String>? = null,
        val source: String? = null,
        val subject: List<String>? = null,
        val subtitle: List<String>? = null,
        val title: List<String>? = null,
        val type: String? = null,
        val volume: String? = null
    )

    data class Query(
        @Json(name="search-terms")
        val search_terms: String? = null,
        @Json(name="start-index")
        val start_index: Int? = null
    )

    data class Auth(
        val affiliation: List<Any>? = null,
        val family: String? = null,
        val given: String? = null,
        val sequence: String? = null
    )

    data class ContentDomain(
        @Json(name="crossmark-restriction")
        val crossmark_restriction: Boolean? = null,
        val domain: List<Any>? = null
    )

    data class Created(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null,
        @Json(name="date-time")
        val date_time: String? = null,
        val timestamp: Long? = null
    )

    data class Deposited(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null,
        @Json(name="date-time")
        val date_time: String? = null,
        val timestamp: Long? = null
    )

    data class Indexed(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null,
        @Json(name="date-time")
        val date_time: String? = null,
        val timestamp: Long? = null
    )

    data class IssnType(
        val type: String? = null,
        val value: String? = null
    )

    data class Issued(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null
    )

    data class JournalIssue(
        val issue: String? = null,
        @Json(name="published-print")
        val published_print: PublishedPrint? = null
    )

    data class License(
        val URL: String? = null,
        @Json(name="content-version")
        val content_version: String? = null,
        @Json(name="delay-in-days")
        val delay_in_days: Int? = null,
        val start: Start? = null
    )

    data class Link(
        val URL: String? = null,
        @Json(name="content-type")
        val content_type: String? = null,
        @Json(name="content-version")
        val content_version: String? = null,
        @Json(name="intended-application")
        val intended_application: String? = null
    )

    data class PublishedOnline(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null
    )

    data class PublishedPrintX(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null
    )

    data class Ref(
        val DOI: String? = null,
        val author: String? = null,
        @Json(name="doi-asserted-by")
        val doi_asserted_by: String? = null,
        @Json(name="first-page")
        val first_page: String? = null,
        @Json(name="journal-title")
        val journal_title: String? = null,
        val key: String? = null,
        val unstructured: String? = null,
        val volume: String? = null,
        val year: String? = null
    )

    data class Relation(
        val cites: List<Any>? = null
    )

    data class PublishedPrint(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null
    )

    data class Start(
        @Json(name="date-parts")
        val date_parts: List<List<Int>>? = null,
        @Json(name="date-time")
        val date_time: String? = null,
        val timestamp: Long? = null
    )
}