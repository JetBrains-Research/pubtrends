## Nature Reviews Reference Clustering

**Current status: 40/40 papers processed, ~90% references mapped**

### Validation

1. **(DONE)** Grouped references file should be valid JSONs.
2. Some grouped references files (29147025, 29213134, 26675821, 27919677, 28853444, 31836872, 31937935, 32042144 and 1 more) contain nulls, references should be validated.
3. Paragraph structure (hierarchy) should be validated somehow (maybe via Nature's website?).
4. Fix inconsistent use of tabs and spaces.

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
	26580716: 96 / 152 references mapped
	26580717: 78 / 91 references mapped
	26656254: 122 / 160 references mapped
	26667849: 86 / 99 references mapped
	26675821: 104 / 123 references mapped
	26678314: 144 / 198 references mapped
	26688349: 51 / 106 references mapped
	26688350: 54 / 105 references mapped
	27677859: 99 / 111 references mapped
	27677860: 112 / 178 references mapped
	27834397: 172 / 200 references mapped
	27834398: 173 / 240 references mapped
	27890914: 182 / 254 references mapped
	27904142: 83 / 101 references mapped
	27916977: 90 / 106 references mapped
	28003656: 140 / 196 references mapped
	28792006: 105 / 126 references mapped
	28852220: 112 / 137 references mapped
	28853444: 75 / 89 references mapped
	28920587: 166 / 188 references mapped
	29147025: 140 / 172 references mapped
	29170536: 127 / 161 references mapped
	29213134: 117 / 151 references mapped
	29321682: 150 / 185 references mapped
	30108335: 231 / 277 references mapped
	30390028: 145 / 162 references mapped
	30459365: 278 / 333 references mapped
	30467385: 146 / 177 references mapped
	30578414: 120 / 127 references mapped
	30644449: 127 / 164 references mapped
	30679807: 127 / 157 references mapped
	30842595: 162 / 209 references mapped
	31686003: 168 / 195 references mapped
	31806885: 166 / 203 references mapped
	31836872: 32 / 79 references mapped
	31937935: 229 / 279 references mapped
	32005979: 109 / 139 references mapped
	32020081: 163 / 178 references mapped
	32042144: 134 / 204 references mapped
	32699292: 136 / 153 references mapped
	```
  
  * other thoughts:
    * 26667849 - timeline of key findings about DNA with references, articles like these might serve as a ground truth for topic evolution analysis
    * 31937935 - very systematic!

### How to reproduce: 

TODO (if needed)
