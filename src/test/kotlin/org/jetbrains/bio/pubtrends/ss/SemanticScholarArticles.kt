package org.jetbrains.bio.pubtrends.ss

// sample articles

object SemanticScholarArticles {
    val articles = listOf(
            SemanticScholarArticle(ssid = "03029e4427cfe66c3da6257979dc2d5b6eb3a0e4",
                    pmid = 2252909,
                    citations = listOf("5451b1ef43678d473575bdfa7016d024146f2b53",
                            "3c61e275f82faa0c3f370d16d4c254bdf592c05c", "9ed11efec96327fd6ad9e346a0b2d3a70d00ba2e"),
                    title = "Primary Debulking Surgery Versus Neoadjuvant Chemotherapy in Stage IV Ovarian Cancer",
                    year = 2011,
                    doi = "10.1245/s10434-011-2100-x",
                    aux = AuxInfo(journal = Journal(name = "Annals of Surgical Oncology", volume = "19", pages = "959-965"),
                            authors = listOf(Author(name = "Jose Alejandro Rauh-Hain")),
                            venue = "Annals of Surgical Oncology",
                            links = Links(s2Url = "https://semanticscholar.org/paper/4cd223df721b722b1c40689caa52932a41fcc223",
                                    pdfUrls = listOf("https://doi.org/10.1093/llc/fqu052")))),


            SemanticScholarArticle(ssid = "4cbba8127c8747a3b2cfb9c1f48c43e5c15e323e",
                    pmid = 7629622,
                    citations = listOf("46ebb8d96613bb0508688a71e0d6ce67c7e3d041",
                            "347259c8a6e3aa72411cf51c4c90e8a0261dd100", "9124c0d40c07e2218cb35355daa2c8c0ae2c6e11"),
                    title = "Lipid transport function of lipoproteins in flying insects.",
                    year = 1990,
                    keywords = "Lipid Transport,Lipoproteins",
                    aux = AuxInfo(journal = Journal(name = "Biochimica et biophysica acta", volume = "1047 3", pages = "195-211"),
                            authors = listOf(Author(name = "Dick J van der Horst")),
                            venue = "Biochimica et biophysica acta")),


            SemanticScholarArticle("58ff17c7d8ca006731facf7771761946350db062",
                    pmid = 567834224,
                    citations = listOf("585c33b4e6b4613e6403ccd16516c335d36ab2c7",
                            "2430c97f1aa9c8036702f8f9686d7056a58775fa", "01829ca43653742e9749122560b1b4866df07bd5",
                            "bd391353a350c3b912592a8159caf55c87d552d3"),
                    title = "Acute spastic entropion.",
                    year = 1976,
                    keywords = "Spastic entropion",
                    aux = AuxInfo(journal = Journal(name = "Canadian journal of ophthalmology. Journal canadien d'ophtalmologie",
                            volume = "11 4", pages = "346"),
                            authors = listOf(Author(name = "R I Noble")),
                            venue = "Canadian journal of ophthalmology. Journal canadien d'ophtalmologie",
                            links = Links(s2Url = "https://semanticscholar.org/paper/34ca6d85db744543ddc27d74d7f225b13c66b95f"))),


            SemanticScholarArticle("2430c97f1aa9c8036702f8f9686d7056a58775fa",
                    pmid = 8548282,
                    title = "Knowledge-rich, computer-assisted composition of Chinese couplets"),


            SemanticScholarArticle("585c33b4e6b4613e6403ccd16516c335d36ab2c7",
                    title = "Article from ArXiv with id and title",
                    aux = AuxInfo(links = Links(pdfUrls = listOf("http://arxiv.org/pdf/cond-mat/0606534v1.pdf")))))
}
