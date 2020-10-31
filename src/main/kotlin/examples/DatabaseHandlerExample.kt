package examples

import org.jetbrains.bio.pubtrends.arxiv.ArxivData
import org.jetbrains.bio.pubtrends.data.Author
import org.jetbrains.bio.pubtrends.data.JournalRef
import org.jetbrains.bio.pubtrends.neo4j.DatabaseHandler
import org.jetbrains.bio.pubtrends.data.Reference

fun main() {
    val dbh = DatabaseHandler("localhost", "7687", "neo4j", "qwerty")
    val records = mutableListOf<ArxivData>()
    val r1 = ArxivData("1")
    r1.id = "1"
    r1.doi = "doi:1"
    r1.title = "Planets"
    r1.authors.add(Author("Author3", "University1"))
    r1.authors.add(Author("Author1"))
    r1.journal = JournalRef("Physics")
    r1.refList.add(Reference("Abracadabra", false).apply { title = "Abracadabra" })
    records.add(r1)
    val r2 = ArxivData("2")
    r2.id = "2"
    r2.title = "Stars"
    r2.authors.add(Author("Author1", "University1"))
    r2.authors.add(Author("Author2", "University2"))
    r2.refList.add(Reference("Planets", false).apply { title = "Planets" })
    records.add(r2)

    dbh.storeArxivData(records)
    dbh.close()
}

