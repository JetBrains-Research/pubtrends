package org.jetbrains.bio.pubtrends.crawler

object Articles {
    private var storage: Map<Int, PubmedArticle>

    var size: Int = 0
        get() {
            return storage.size
        }

    operator fun get(pmid: Int): PubmedArticle? {
        return storage[pmid]
    }

    init {
        val article29736257 = PubmedArticle(29736257)
        article29736257.year = 2017
        article29736257.title = "Evidence that S6K1, but not 4E-BP1, mediates skeletal muscle " +
                "pathology associated with loss of A-type lamins."
        article29736257.abstractText = "The mechanistic target of rapamycin (mTOR) signaling pathway plays a central " +
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
                "muscle function and can be phenocopied by reduced S6K1 activity, but not 4E-BP1 activation."
        article29736257.keywordList.addAll(listOf("4E-BP1", "Lmna−/− mice", "S6K1", "lifespan",
                "mTORC1", "muscle", "rapamycin"))
        article29736257.citationList.addAll(listOf(10587585, 11855819, 26614871, 12702809))

        val article29456534 = PubmedArticle(29456534)
        article29456534.year = 2018
        article29456534.title = "Critical Role of TGF-β and IL-2 Receptor Signaling in Foxp3 Induction by an " +
                "Inhibitor of DNA Methylation."
        article29456534.abstractText = "Under physiological conditions, CD4+ regulatory T (Treg) cells expressing the" +
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
                "phenotypically and functionally stable iTreg cells."
        article29456534.keywordList.addAll(listOf("CNS2", "DNA methylation", "Foxp3", "epigenetic regulation", "iTreg"))
        article29456534.citationList.addAll(listOf(27989104, 17591856, 25159127, 23123060, 22343569))

        val article20453483 = PubmedArticle(20453483)
        article20453483.year = 2011
        article20453483.title = "Coping strategies used by seniors going through the normal aging process: " +
                "does fear of falling matter?"
        article20453483.abstractText = "Recent studies show that fear of falling, a frequent fear of " +
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

        val article27654823 = PubmedArticle(27654823)
        article27654823.year = 2017
        article27654823.title = "Production of 10S-hydroxy-8(E)-octadecenoic acid from oleic acid by whole " +
                "recombinant Escherichia coli cells expressing 10S-dioxygenase from Nostoc punctiforme PCC 73102 " +
                "with the aid of a chaperone."
        article27654823.abstractText = "To increase the production of 10S-hydroxy-8(E)-octadecenoic acid from oleic " +
                "acid by whole recombinant Escherichia coli cells expressing Nostoc punctiforme 10S-dioxygenase with " +
                "the aid of a chaperone. The optimal conditions for 10S-hydroxy-8(E)-octadecenoic acid production by " +
                "recombinant cells co-expressing chaperone plasmid were pH 9, 35 °C, 15 % (v/v) dimethyl sulfoxide, " +
                "40 g cells l-1, and 10 g oleic acid l-1. Under these conditions, recombinant cells co-expressing " +
                "chaperone plasmid produced 7.2 g 10S-hydroxy-8(E)-octadecenoic acid l-1 within 30 min, with a " +
                "conversion yield of 72 % (w/w) and a volumetric productivity of 14.4 g l-1 h-1. The activity of " +
                "recombinant cells expressing 10S-dioxygenase was increased by 200 % with the aid of a chaperone, " +
                "demonstrating the first biotechnological production of 10S-hydroxy-8(E)-octadecenoic acid using " +
                "recombinant cells expressing 10S-dioxygenase."
        article27654823.keywordList.addAll(listOf("10S-Dioxygenase", "10S-Hydroxy-8(E)-octadecenoic acid",
                "Biotransformation", "Chaperone", "Nostoc punctiforme PCC 73102", "Oleic acid"))

        storage = mapOf(29736257 to article29736257,
                29456534 to article29456534,
                20453483 to article20453483,
                27654823 to article27654823)
    }
}