## Nature Reviews Reference Clustering

**Current status: 20/40 papers processed, ~76% references mapped**

This folder contains:
 * `clustering/` - the final result of preprocessing for each paper
 * `grouped_refs_validated` - hand-curated mapping of references by section of the paper
 
Important things:
 * clustering contains PMIDs of references grouped by paper sections in the exact order, which means:
   * PMIDs within a cluster may repeat
   * one PMID may occur in several clusters
 * there are special kinds of sections, which can be treated in a different way, see titles of these sections below:
   * INTRODUCTION
   * CONCLUSION
   * PERSPECTIVES
   * Box N | Title goes here
 * not all references are currently mapped due to Grobid parsing errors, hopefully, will be fixed in the near future, details below:

	```
	30108335: 231 / 277 references mapped
	28792006: 105 / 126 references mapped
	30390028: 145 / 162 references mapped
	29213134: 117 / 151 references mapped
	26580716: 96 / 152 references mapped
	26688350: 54 / 105 references mapped
	27677860: 112 / 178 references mapped
	27834398: 173 / 240 references mapped
	28852220: 112 / 137 references mapped
	29147025: 140 / 172 references mapped
	26688349: 51 / 106 references mapped
	27904142: 83 / 101 references mapped
	28003656: 140 / 196 references mapped
	29321682: 150 / 185 references mapped
	29170536: 127 / 161 references mapped
	27677859: 99 / 111 references mapped
	26656254: 122 / 160 references mapped
	26678314: 144 / 198 references mapped
	27834397: 172 / 200 references mapped
	26580717: 78 / 91 references mapped
	```

How to reproduce: TODO (if needed)