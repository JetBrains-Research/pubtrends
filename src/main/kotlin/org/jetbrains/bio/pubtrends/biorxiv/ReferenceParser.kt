package org.jetbrains.bio.pubtrends.biorxiv

import com.google.gson.Gson
import org.jetbrains.bio.pubtrends.pm.PubmedArticle
import java.io.BufferedReader
import java.io.FileReader
import java.io.InputStreamReader
import java.nio.file.Path

class ReferenceParser(
//        private val dbHandler: BiorxivNeo4jDatabaseHandler,
        private val referencesPath: Path,
        private val anystylePath: Path
) {
    fun parse(references: List<String>): List<PubmedArticle> {
        referencesPath.toFile().outputStream().bufferedWriter().use {
            references.forEach {
                ref -> it.write("${ref.trim('\n')}\n")
            }
        }
        runAnystyleParser()

        return processAnystyleOutput().map {
            it.toPubmedArticle()
        }
//
//        // Attempt to find all references in Pubmed, then drop all zeros (references that were not found)
//        return parsedReferences.map { dbHandler.getPMID(it) }.filter { it > 0 }
    }

    fun runAnystyleParser() : Boolean {
        val commands = listOf("anystyle", "--stdout", "-f", "json", referencesPath.toAbsolutePath())
        val process = Runtime.getRuntime().exec(commands.joinToString { " " })

        val reader = BufferedReader(InputStreamReader(process.inputStream))
        val contents = reader.readLines()

        return true
    }

    fun processAnystyleOutput() : List<AnystyleReference> {
        val contents = BufferedReader(FileReader(anystylePath.toFile())).use {
            it.readLines().joinToString("\n")
        }

        // Minor wrapping of anystyle output to be a valid JSON
        val wrappedContents = "{\"references\": $contents }"
        println(wrappedContents)

        val anystyleData = Gson().fromJson(wrappedContents, AnystyleData::class.java)
        return anystyleData.references
    }
}