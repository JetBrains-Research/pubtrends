package org.jetbrains.bio.pubtrends.pm

private val PMA1 = PubmedArticle(
    pmid = 1, title = "Article Title 1", year = 1963, doi = "10.000/0000",
    aux = AuxInfo(
        authors = listOf(Author(name = "Geller R"), Author(name = "Geller M"), Author(name = "Bing Ch")),
        journal = Journal(name = "Nature")
    )
)
private val PMA2 = PubmedArticle(
    2, "Article Title 2", abstract = "Abstract", year = 1965,
    aux = AuxInfo(
        authors = listOf(Author(name = "Buffay Ph"), Author(name = "Geller M"), Author(name = "Doe J")),
        journal = Journal(name = "Science")
    )
)
private val PMA3 = PubmedArticle(
    3, "Article Title 3", abstract = "Other Abstract", year = 1967,
    aux = AuxInfo(
        authors = listOf(Author(name = "Doe J"), Author(name = "Buffay Ph")),
        journal = Journal(name = "Nature")
    )
)
private val PMA4 = PubmedArticle(
    4, "Article Title 4", year = 1968,
    aux = AuxInfo(
        authors = listOf(Author(name = "Doe J"), Author(name = "Geller R")),
        journal = Journal(name = "Science")
    )
)
private val PMA5 = PubmedArticle(
    5, "Article Title 5", year = 1975,
    aux = AuxInfo(
        authors = listOf(Author(name = "Green R"), Author(name = "Geller R"), Author(name = "Doe J")),
        journal = Journal(name = "Nature")
    )
)
private val PMA6 = PubmedArticle(6, "Article Title 6", type = PublicationType.Review)

val PM_REQUIRED_ARTICLES = listOf(
    PMA1, PMA2, PMA3, PMA4, PMA5, PMA6
)

val PM_EXTRA_ARTICLES = listOf(
    PubmedArticle(7, "Article Title 7", year = 1968),
    PubmedArticle(8, "Article Title 8", year = 1969),
    PubmedArticle(9, "Article Title 9", year = 1970),
    PubmedArticle(10, "Article Title 10", year = 1970)
)

val PM_EXTRA_ARTICLE = PubmedArticle(
    pmid = 100,
    title = "activemodule dna",
    abstract = """
    ... physical activity module ... (inactive) ... activating                modules ... antipositive ... active ...
... methylation ...
""",
    year = 2021,
    doi = "10.1249/01.mss.0000229457.73333.9a,doi:10.1101/gad.918201 ",
    aux = AuxInfo(journal = Journal(name = ""))
)

val PM_INNER_CITATIONS = listOf(
    (2 to 1), (3 to 2), (4 to 3), (5 to 4), (6 to 5)
)

val PM_OUTER_CITATIONS = listOf(
    (7 to 1), (7 to 2), (7 to 3), (8 to 1), (8 to 3),
    (8 to 4), (9 to 4), (9 to 5), (10 to 4), (10 to 5)
)
