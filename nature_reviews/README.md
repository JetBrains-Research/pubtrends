## Nature Reviews Reference Clustering

**Current status: 40/40 papers processed, ~96% references mapped**

### Validation

1. **(DONE)** Grouped references file should be valid JSONs.
2. **(DONE)** Grouped references files should not contain nulls, reference IDs should be unique.
3. References should be validated.
4. Paragraph structure (hierarchy) should be validated somehow (maybe via Nature's website?).
5. Fix inconsistent use of tabs and spaces.

### Description

This folder contains:
 * `src/collect_papers.sh` - source code for fetching appropriate review papers with information about references
 * `clustering/` - the final result of preprocessing for each paper
 * `grouped_refs_validated` - hand-curated mapping of references by section of the paper
 * `refs_validated` - files with references that were originally extracted by Grobid and later partially fixed to get better mapping with Pubmed titles
 
Important things:
 * clustering contains PMIDs of references grouped by paper sections in the exact order, which means:
   * PMIDs within a cluster may repeat
   * one PMID may occur in several clusters
 * there are special kinds of sections, which can be treated in a different way, see titles of these sections below:
   * INTRODUCTION
   * CONCLUSION
   * PERSPECTIVES
   * Box N | Title goes here
 * sections without references are deleted from the markup, same for figure captions
 * several papers might contain clustering-related info in tables, not processed currently:
   * 29147025 - tables 1 and 2 with references
   * 27677859 - table may contain a good clustering (p. 46)
   * 27677860 - structured conclusion
   * 29213134 - interesting review about attention
   * 26667849 - timeline of key findings about DNA with references
   * 27890914, 28920587, 30467385, 30679807, 30842595, 31806885, 32005979, 32020081 - another structured table
   * 31937935 - systematic!
 * not all references are currently mapped due to Grobid parsing errors, hopefully, will be fixed in the near future, details below:

	```
	26580716: 150 / 152 references mapped
	26580717: 86 / 91 references mapped
	26656254: 147 / 160 references mapped
	26667849: 94 / 99 references mapped
	26675821: 120 / 123 references mapped
	26678314: 189 / 198 references mapped
	26688349: 101 / 106 references mapped
	26688350: 100 / 105 references mapped
	27677859: 105 / 111 references mapped
	27677860: 169 / 178 references mapped
	27834397: 191 / 200 references mapped
	27834398: 198 / 240 references mapped
	27890914: 246 / 254 references mapped
	27904142: 93 / 101 references mapped
	27916977: [100%] 106 / 106 references mapped
	28003656: 166 / 196 references mapped
	28792006: 117 / 126 references mapped
	28852220: 128 / 137 references mapped
	28853444: 88 / 89 references mapped
	28920587: 179 / 188 references mapped
	29147025: 160 / 172 references mapped
	29170536: 149 / 161 references mapped
	29213134: 136 / 151 references mapped
	29321682: 179 / 185 references mapped
	30108335: 268 / 277 references mapped
	30390028: 160 / 162 references mapped
	30459365: 310 / 333 references mapped
	30467385: 174 / 177 references mapped
	30578414: 123 / 127 references mapped
	30644449: 153 / 164 references mapped
	30679807: 142 / 157 references mapped
	30842595: 197 / 209 references mapped
	31686003: 187 / 195 references mapped
	31806885: 195 / 203 references mapped
	31836872: 37 / 79 references mapped
	31937935: 266 / 279 references mapped
	32005979: 129 / 139 references mapped
	32020081: 174 / 178 references mapped
	32042144: 144 / 204 references mapped
	32699292: 146 / 153 references mapped
	```
  
  * other thoughts:
    * 26667849 - timeline of key findings about DNA with references, articles like these might serve as a ground truth for topic evolution analysis
    * 31937935 - very systematic!

### How to reproduce: 

TODO (if needed)
