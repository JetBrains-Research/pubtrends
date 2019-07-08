package org.jetbrains.bio.pubtrends.ssprocessing

import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.Table


class SemanticScholarVerifier {
    companion object {
        private val logger = LogManager.getLogger(SemanticScholarVerifier::class)
    }

    fun verify(first: Table, second: Table) {
        // cant find sql "except" operator at "exposed"
        val sqlExceptStatement = """
                SELECT COUNT(*) FROM
                (SELECT pmid_citing, pmid_cited
                FROM PmidCitationsFromSS
                EXCEPT
                SELECT pmid_citing, pmid_cited
                FROM Citations) AS Diff;
                """

        logger.info("If you want to check how many citations from parsed file are not contained in PubMed database:")
        logger.info("Run \"psql pubmed\" and the following command:")
        logger.info(sqlExceptStatement)
        // in progress..
    }
}
