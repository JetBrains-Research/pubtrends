package org.jetbrains.bio.pubtrends.pm

import org.joda.time.DateTime

object PubmedArticles {
    val articles = mapOf(
        420880 to PubmedArticle(
            pmid = 420880,
            date = DateTime(1979, 1, 1, 12, 0),
            title = "Changes in DNA methylation in rat during ontogenesis and under effects of hydrocortisone",
            type = PublicationType.Article,
            meshHeadingList = listOf(
                "Aging", "Animals", "Animals Newborn", "Brain growth & development metabolism",
                "DNA metabolism", "DNA (Cytosine-5-)-Methyltransferases metabolism", "Embryo Mammalian",
                "Female", "Hydrocortisone pharmacology", "Kinetics", "Liver growth & development metabolism",
                "Methylation", "Methyltransferases metabolism", "Pregnancy", "Rats"
            ),
            auxInfo = ArticleAuxInfo(
                databanks = listOf(
                    DatabankEntry(
                        name = "GENBANK",
                        accessionNumber = listOf("AF321191", "AF321192")
                    ),
                    DatabankEntry(
                        name = "OMIM",
                        accessionNumber = listOf("118200", "145900", "162500", "605253")
                    )
                ),
                journal = Journal("Biokhimiia (Moscow, Russia)"),
                language = "rus"
            )
        ),

        29736257 to PubmedArticle(
            pmid = 29736257,
            date = DateTime(2017, 1, 1, 12, 0),
            title = "Evidence that S6K1, but not 4E-BP1, mediates skeletal muscle " +
                    "pathology associated with loss of A-type lamins",
            abstractText = "The mechanistic target of rapamycin (mTOR) signaling pathway plays a central " +
                    "role in aging and a number of different disease states. Rapamycin, which suppresses activity" +
                    " of the mTOR complex 1 (mTORC1), shows preclinical (and sometimes clinical) efficacy in a" +
                    " number of disease models. Among these are Lmna-/- mice, which serve as a mouse model " +
                    "for dystrophy-associated laminopathies. To confirm that elevated mTORC1 signaling is responsible " +
                    "for the pathology manifested in Lmna-/- mice and to decipher downstream genetic mechanisms " +
                    "underlying the benefits of rapamycin, we tested in Lmna-/- mice whether survival could be extended " +
                    "and disease pathology suppressed either by reduced levels of S6K1 or enhanced levels of 4E-BP1, " +
                    "two canonical mTORC1 substrates. Global heterozygosity for S6K1 ubiquitously extended lifespan of " +
                    "Lmna-/- mice (Lmna-/-S6K1+/- mice). This life extension is due to improving muscle, but not heart " +
                    "or adipose, function, consistent with the observation that genetic ablation of S6K1 specifically " +
                    "in muscle tissue also extended survival of Lmna-/- mice. In contrast, whole-body overexpression of " +
                    "4E-BP1 shortened the survival of Lmna-/- mice, likely by accelerating lipolysis. Thus, " +
                    "rapamycin-mediated lifespan extension in Lmna-/- mice is in part due to the improvement of skeletal " +
                    "muscle function and can be phenocopied by reduced S6K1 activity, but not 4E-BP1 activation.",
            keywordList = listOf(
                "4E-BP1", "Lmna−/− mice", "S6K1", "lifespan",
                "mTORC1", "muscle", "rapamycin"
            ),
            citationList = listOf(10587585, 11855819, 26614871, 12702809)
        ),

        29456534 to PubmedArticle(
            pmid = 29456534,
            date = DateTime(2018, 1, 1, 12, 0),
            title = "Critical Role of TGF-β and IL-2 Receptor Signaling in Foxp3 Induction by an " +
                    "Inhibitor of DNA Methylation",
            abstractText = "Under physiological conditions, CD4+ regulatory T (Treg) cells expressing the" +
                    " transcription factor Foxp3 are generated in the thymus [thymus-derived Foxp3+ Treg (tTregs) cells] " +
                    "and extrathymically at peripheral sites [peripherally induced Foxp3+ Treg (pTreg) cell], and both " +
                    "developmental subsets play non-redundant roles in maintaining self-tolerance throughout life. In " +
                    "addition, a variety of experimental in vitro and in vivo modalities can extrathymically elicit a " +
                    "Foxp3+ Treg cell phenotype in peripheral CD4+Foxp3- T cells, which has attracted much interest as " +
                    "an approach toward cell-based therapy in clinical settings of undesired immune responses. A " +
                    "particularly notable example is the in vitro induction of Foxp3 expression and Treg cell activity " +
                    "(iTreg cells) in initially naive CD4+Foxp3- T cells through T cell receptor (TCR) and IL-2R " +
                    "ligation, in the presence of exogenous TGF-β. Clinical application of Foxp3+ iTreg cells has been " +
                    "hampered by the fact that TGF-β-driven Foxp3 induction is not sufficient to fully recapitulate the " +
                    "epigenetic and transcriptional signature of in vivo induced Foxp3+ tTreg and pTreg cells, which " +
                    "includes the failure to imprint iTreg cells with stable Foxp3 expression. This hurdle can be " +
                    "potentially overcome by pharmacological interference with DNA methyltransferase activity and CpG " +
                    "methylation [e.g., by the cytosine nucleoside analog 5-aza-2'-deoxycytidine (5-aza-dC)] to " +
                    "stabilize TGF-β-induced Foxp3 expression and to promote a Foxp3+ iTreg cell phenotype even in the " +
                    "absence of added TGF-β. However, the molecular mechanisms of 5-aza-dC-mediated Foxp3+ iTreg cell " +
                    "generation have remained incompletely understood. Here, we show that in the absence of exogenously " +
                    "added TGF-β and IL-2, efficient 5-aza-dC-mediated Foxp3+ iTreg cell generation from TCR-stimulated " +
                    "CD4+Foxp3- T cells is critically dependent on TGF-βR and IL-2R signaling and that this process is " +
                    "driven by TGF-β and IL-2, which could either be FCS derived or produced by T cells on TCR " +
                    "stimulation. Overall, these findings contribute to our understanding of the molecular mechanisms " +
                    "underlying the process of Foxp3 induction and may provide a rational basis for generating " +
                    "phenotypically and functionally stable iTreg cells.",
            keywordList = listOf("CNS2", "DNA methylation", "Foxp3", "epigenetic regulation", "iTreg"),
            citationList = listOf(27989104, 17591856, 25159127, 23123060, 22343569),
            auxInfo = ArticleAuxInfo(
                authors = listOf(
                    Author(
                        name = "Freudenberg K",
                        affiliation = listOf(
                            "Molecular and Cellular Immunology/Immune Regulation, " +
                                    "DFG-Center for Regenerative Therapies Dresden (CRTD), Center for Molecular and Cellular" +
                                    " Bioengineering (CMCB), Technische Universität Dresden, Dresden, Germany"
                        )
                    ),
                    Author(
                        name = "Dohnke S",
                        affiliation = listOf(
                            "Molecular and Cellular Immunology/Immune Regulation, " +
                                    "DFG-Center for Regenerative Therapies Dresden (CRTD), Center for Molecular and Cellular" +
                                    " Bioengineering (CMCB), Technische Universität Dresden, Dresden, Germany",
                            "Osteoimmunology," +
                                    " DFG-Center for Regenerative Therapies Dresden (CRTD), Center for Molecular and Cellular " +
                                    "Bioengineering (CMCB), Technische Universität Dresden, Dresden, Germany"
                        )
                    )
                )
            )
        ),

        20453483 to PubmedArticle(
            pmid = 20453483,
            date = DateTime(2011, 1, 1, 12, 0),
            title = "Coping strategies used by seniors going through the normal aging process: " +
                    "does fear of falling matter?",
            abstractText = "Recent studies show that fear of falling, a frequent fear of " +
                    "community-dwelling seniors, can have a negative impact on their health and quality of life. When " +
                    "fear of falling is intense, it can prompt individuals to limit or avoid certain activities. This " +
                    "activity restriction can lead to premature physical and functional decline and, ultimately, increase" +
                    " the risk for falls. Although activity avoidance/restriction is a common strategy used by seniors to" +
                    " cope with fear of falling, they may use other strategies as well to cope with this fear. However, " +
                    "these other strategies have received little attention to date. This study aimed at examining and " +
                    "comparing coping strategies used by seniors with and without fear of falling. It also examined if " +
                    "fear of falling is an independent correlate of the use of coping strategies among seniors. 288 " +
                    "seniors aged 65 years or over and going through the normal aging process were assessed during " +
                    "structured home interviews. Fear of falling was assessed through a single question (Are you afraid " +
                    "of falling?) and a 4-category response scale (never, occasionally, often, very often). Coping " +
                    "strategies used by participants were assessed with the Inventory of Coping Strategies Used by the " +
                    "Elderly. Findings show that seniors with fear of falling use several coping strategies other than " +
                    "activity avoidance/restriction in their daily functioning. Compared with nonfearful seniors, they " +
                    "tend to use a wider range of coping strategies and use them more frequently. Results also indicate " +
                    "that fear of falling is an independent correlate of diversity and frequency of use of behavioral " +
                    "coping strategies. This study suggests that fall prevention practitioners and researchers should " +
                    "document the range and frequency of use of strategies that seniors may employ to cope with fear of " +
                    "falling. These data could help improve interventions and evaluative research in the domain of fall " +
                    "prevention."
        ),

        27654823 to PubmedArticle(
            pmid = 27654823,
            date = DateTime(2017, 1, 1, 12, 0),
            title = "Production of 10S-hydroxy-8(E)-octadecenoic acid from oleic acid by whole " +
                    "recombinant Escherichia coli cells expressing 10S-dioxygenase from Nostoc punctiforme PCC 73102 " +
                    "with the aid of a chaperone",
            abstractText = "To increase the production of 10S-hydroxy-8(E)-octadecenoic acid from oleic " +
                    "acid by whole recombinant Escherichia coli cells expressing Nostoc punctiforme 10S-dioxygenase with " +
                    "the aid of a chaperone. The optimal conditions for 10S-hydroxy-8(E)-octadecenoic acid production by " +
                    "recombinant cells co-expressing chaperone plasmid were pH 9, 35 °C, 15 % (v/v) dimethyl sulfoxide, " +
                    "40 g cells l-1, and 10 g oleic acid l-1. Under these conditions, recombinant cells co-expressing " +
                    "chaperone plasmid produced 7.2 g 10S-hydroxy-8(E)-octadecenoic acid l-1 within 30 min, with a " +
                    "conversion yield of 72 % (w/w) and a volumetric productivity of 14.4 g l-1 h-1. The activity of " +
                    "recombinant cells expressing 10S-dioxygenase was increased by 200 % with the aid of a chaperone, " +
                    "demonstrating the first biotechnological production of 10S-hydroxy-8(E)-octadecenoic acid using " +
                    "recombinant cells expressing 10S-dioxygenase.",
            keywordList = listOf(
                "10S-Dioxygenase", "10S-Hydroxy-8(E)-octadecenoic acid",
                "Biotransformation", "Chaperone", "Nostoc punctiforme PCC 73102", "Oleic acid"
            )
        ),

        11243089 to PubmedArticle(
            pmid = 11243089,
            date = DateTime(1998, 1, 1, 12, 0),
            title = "Nutritional status of pavement dweller children of Calcutta City",
            abstractText = "Pavement dwelling is likely to aggravate malnutrition among its residents due " +
                    "to extreme poverty, lack of dwelling and access to food and their exposure to polluted environment. " +
                    "Paucity of information about nutritional status of street children compared to that among urban " +
                    "slum dwellers, squatters or rural/tribal population is quite evident. The present study revealed " +
                    "the magnitude of Protein Energy Malnutrition (PEM) and few associated factors among a sample of " +
                    "435 underfives belonging to pavement dweller families and selected randomly from clusters of " +
                    "such families, from each of the five geographical sectors of Calcutta city. Overall prevalence " +
                    "of PEM was found almost similar (about 70%) to that among other 'urban poor' children viz. " +
                    "slum dwellers etc., but about 16% of them were found severely undernourished (Grade III & " +
                    "V of IAP classification of PEM). About 35% and 70% of street dweller children had wasting and " +
                    "stunting respectively. Severe PEM (Grade III & IV) was more prevalent among 12-23 months " +
                    "old, girl child, those belonged to illiterate parents and housewife mothers rather than wage " +
                    "earners. It also did increase with increase of birth rate of decrease of birth interval. This document " +
                    "presents a cross-sectional survey concerning the magnitude of protein energy malnutrition (PEM) and " +
                    "its associated factors among 435 under-5 pavement-dwelling children in Calcutta. Results revealed " +
                    "that 69.43% were undernourished and that 16% of them were suffering from severe malnutrition (grade " +
                    "III and IV of the Indian Academy of Pediatrics criteria for PEM). The 24-35 month age group had the " +
                    "highest prevalence of malnutrition (82.93%) followed by the 36-47 and 12-23 month age groups with " +
                    "prevalences of 76.19% and 74.03%, respectively. Prevalence of severe grade malnutrition was noted to " +
                    "be three times higher in females (24.76%) than males (8.45%), and among families it increased in " +
                    "direct proportion to birth rate and inverse proportion to birth interval. Moreover, children of " +
                    "illiterate parents and nonworking mothers had a higher incidence of severe PEM. Simple measures " +
                    "such as exclusive breast-feeding and timely complementary feeding as well as measures directed " +
                    "toward birth spacing and limiting family size should be implemented to solve the problem of malnutrition.",
            keywordList = listOf(
                "Age Factors", "Child Nutrition", "Geographic Factors",
                "Population Characteristics", "Spatial Distribution", "Urban Population"
            ),
            type = PublicationType.ClinicalTrial
        ),

        11540070 to PubmedArticle(
            pmid = 11540070,
            date = DateTime(1987, 1, 1, 12, 0),
            title = "Calcium messenger system in plants",
            abstractText = "The purpose of this review is to delineate the ubiquitous and pivotal role " +
                    "of Ca2+ in diverse physiological processes. Emphasis will be given to the role of Ca2+ in " +
                    "stimulus-response coupling. In addition to reviewing the present status of research, our intention " +
                    "is to critically evaluate the existing data and describe the newly developing areas of Ca2+ " +
                    "research in plants.",
            keywordList = listOf(
                "NASA Discipline Number 40-30", "NASA Discipline Plant Biology",
                "NASA Program Space Biology", "Non-NASA Center"
            ),
            type = PublicationType.Review
        ),

        10188493 to PubmedArticle(
            pmid = 10188493,
            date = DateTime(1998, 12, 1, 12, 0),
            title = "Women's health osteopathy: an alternative view"
        ),

        14316043 to PubmedArticle(
            pmid = 14316043,
            date = DateTime(1965, 12, 1, 12, 0),
            title = "THE RESPONSIBILITY OF THE DENTIST AND THE DENTAL PROFESSION WITH RESPECT TO JAW " +
                    "FRACTURES",
            keywordList = listOf(
                "DENTISTS", "FRACTURE FIXATION", "FRACTURES",
                "INTERPROFESSIONAL RELATIONS", "JAW", "MANDIBULAR INJURIES", "MAXILLOFACIAL INJURIES",
                "PRACTICE MANAGEMENT DENTAL"
            )
        ),

        18122624 to PubmedArticle(
            pmid = 18122624,
            date = DateTime(1947, 1, 1, 12, 0),
            title = "Mesenteric vascular occlusion",
            keywordList = listOf("MESENTERY occlusion")
        ),

        24884411 to PubmedArticle(
            pmid = 24884411,
            date = DateTime(2014, 5, 12, 12, 0),
            title = "A multilocus timescale for oomycete evolution estimated under three distinct " +
                    "molecular clock models",
            abstractText = "Molecular clock methodologies allow for the estimation of divergence times " +
                    "across a variety of organisms; this can be particularly useful for groups lacking robust fossil " +
                    "histories, such as microbial eukaryotes with few distinguishing morphological traits. Here we have" +
                    " used a Bayesian molecular clock method under three distinct clock models to estimate divergence " +
                    "times within oomycetes, a group of fungal-like eukaryotes that are ubiquitous in the environment " +
                    "and include a number of devastating pathogenic species. The earliest fossil evidence for oomycetes" +
                    " comes from the Lower Devonian (~400 Ma), however the taxonomic affinities of these fossils " +
                    "are unclear. Complete genome sequences were used to identify orthologous proteins among oomycetes, " +
                    "diatoms, and a brown alga, with a focus on conserved regulators of gene expression such as DNA and " +
                    "histone modifiers and transcription factors. Our molecular clock estimates place the origin of " +
                    "oomycetes by at least the mid-Paleozoic (~430-400 Ma), with the divergence between two major " +
                    "lineages, the peronosporaleans and saprolegnialeans, in the early Mesozoic (~225-190 Ma). Divergence" +
                    " times estimated under the three clock models were similar, although only the strict and random " +
                    "local clock models produced reliable estimates for most parameters. Our molecular timescale suggests" +
                    " that modern pathogenic oomycetes diverged well after the origin of their respective hosts, " +
                    "indicating that environmental conditions or perhaps horizontal gene transfer events, rather than" +
                    " host availability, may have driven lineage diversification. Our findings also suggest that the " +
                    "last common ancestor of oomycetes possessed a full complement of eukaryotic regulatory proteins, " +
                    "including those involved in histone modification, RNA interference, and tRNA and rRNA methylation; " +
                    "interestingly no match to canonical DNA methyltransferases could be identified in the oomycete " +
                    "genomes studied here.",
            citationList = listOf(
                23785293, 16822745, 23634808, 19487243, 22127870, 16946064,
                19741609, 12396585, 20862282, 16381920, 20093431, 23020233, 16765584, 19713749, 15952895, 18092388,
                22712506, 21810989, 19158785, 21546353, 21726377, 10198636, 21750662, 21878562, 22803798, 15459382,
                16136655, 21935414, 18024004, 16683862, 21865245, 22367748, 18923393, 22920560, 18715673, 19582169,
                21148394, 20525591, 21424613, 24726347, 20843846, 22105867, 21616882, 18692373, 18705878, 9866200,
                11752195, 20807414, 20626842, 17726520, 17846036, 18451057, 20520714, 21289104
            )
        ),

        29391692 to PubmedArticle(
            pmid = 29391692,
            date = DateTime(2017, 12, 1, 12, 0),
            title = "Partial purification and characterization of glutathione S-transferase from the " +
                    "somatic tissue of Gastrothylax crumenifer (Trematoda: Digenea)",
            doi = "10.14202/vetworld.2017.1493-1500"
        )
    )
}