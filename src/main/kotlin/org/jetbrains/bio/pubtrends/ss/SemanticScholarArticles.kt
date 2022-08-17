package org.jetbrains.bio.pubtrends.ss

private val SS_ARTICLE1 = SemanticScholarArticle(
    ssid = "5451b1ef43678d473575bdfa7016d024146f2b53", crc32id = -410264312,
    title = "I can find this using full text search",
    year = 1999, doi = "10.000/0000"
)

private val SS_ARTICLE2 = SemanticScholarArticle(
    ssid = "cad767094c2c4fff5206793fd8674a10e7fba3fe", crc32id = 984465402,
    title = "Can find using search.", abstract = "Abstract 1",
    year = 1980
)

private val SS_ARTICLE3 = SemanticScholarArticle(
    ssid = "e7cdbddc7af4b6138227139d714df28e2090bd5f", crc32id = 17079054,
    title = "Use search to find it"
)

private val SS_ARTICLE4 = SemanticScholarArticle(
    ssid = "3cf82f53a52867aaade081324dff65dd35b5b7eb", crc32id = -1875049083,
    title = "Want to find it? Just search", year = 1976
)

private val SS_ARTICLE5 = SemanticScholarArticle(
    ssid = "5a63b4199bb58992882b0bf60bc1b1b3f392e5a5", crc32id = 1831680518, pmid = 1,
    title = "Search is key to find", abstract = "Abstract 4",
    year = 2003
)

private val SS_ARTICLE6 = SemanticScholarArticle(
    ssid = "7dc6f2c387193925d3be92d4cc31c7a7cea66d4e", crc32id = -1626578460, pmid = 2,
    title = "Article 6 is here", abstract = "Abstract 6",
    year = 2018
)

private val SS_ARTICLE7 = SemanticScholarArticle(
    ssid = "0f9c1d2a70608d36ad7588d3d93ef261d1ae3203", crc32id = 1075821748, pmid = 3,
    title = "Article 7 is here", abstract = "Abstract 7",
    year = 2010
)

private val SS_ARTICLE8 = SemanticScholarArticle(
    ssid = "872ad0e120b9eefd334562149c065afcfbf90268", crc32id = -1861977375, pmid = 4,
    title = "Article 8 is here", abstract = "Abstract 8",
    year = 1937
)

private val SS_ARTICLE9 = SemanticScholarArticle(
    ssid = "89ffce2b5da6669f63c99ff6398b312389c357dc", crc32id = -1190899769, pmid = 5,
    title = "Article 9 is here", abstract = "Abstract 9"
)

private val SS_ARTICLE10 = SemanticScholarArticle(
    ssid = "390f6fbb1f25bfbc53232e8248c581cdcc1fb9e9", crc32id = -751585733, pmid = 6,
    title = "Article 10 is here", abstract = "Abstract 10",
    year = 2017
)

val SS_REQUIRED_ARTICLES = listOf(
    SS_ARTICLE1, SS_ARTICLE2, SS_ARTICLE3, SS_ARTICLE4, SS_ARTICLE6, SS_ARTICLE7, SS_ARTICLE8, SS_ARTICLE9, SS_ARTICLE10
)

val SS_EXTRA_ARTICLES = listOf(SS_ARTICLE5)

val SS_REQUIRED_CITATIONS = listOf(
    (SS_ARTICLE1.ssid to SS_ARTICLE4.ssid), (SS_ARTICLE1.ssid to SS_ARTICLE3.ssid), (SS_ARTICLE1.ssid to SS_ARTICLE8.ssid),
    (SS_ARTICLE3.ssid to SS_ARTICLE8.ssid), (SS_ARTICLE2.ssid to SS_ARTICLE4.ssid), (SS_ARTICLE2.ssid to SS_ARTICLE3.ssid),
    (SS_ARTICLE6.ssid to SS_ARTICLE7.ssid), (SS_ARTICLE6.ssid to SS_ARTICLE10.ssid)
)
val SS_EXTRA_CITATIONS = listOf((SS_ARTICLE5.ssid to SS_ARTICLE1.ssid))