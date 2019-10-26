package org.jetbrains.bio.pubtrends

import org.neo4j.graphdb.DependencyResolver
import org.neo4j.graphdb.factory.GraphDatabaseFactory
import org.neo4j.graphdb.factory.GraphDatabaseSettings
import org.neo4j.kernel.configuration.BoltConnector
import org.neo4j.kernel.impl.proc.Procedures
import org.neo4j.kernel.internal.GraphDatabaseAPI

object EmbeddedNeo4jInstance {
    // Load configuration
    private val config = Config.config

    private val bolt = BoltConnector("0")
    private val databaseDirectory = createTempDir()

    private val graphDb = GraphDatabaseFactory()
            .newEmbeddedDatabaseBuilder(databaseDirectory.absoluteFile)
            .setConfig(bolt.type, "BOLT")
            .setConfig(bolt.enabled, "true")
            .setConfig(bolt.listen_address, "${config["test_neo4jurl"]}:${config["test_neo4jport"]}")
            .setConfig(GraphDatabaseSettings.procedure_unrestricted,"apoc.*")
            .newGraphDatabase()

    fun load() {
        registerShutdownHook()
        registerProcedures()

        println("Database loaded: ${graphDb.isAvailable(5L)}")
    }

    /**
     * Register APOC procedures to use them in embedded Neo4j database.
     */
    private fun registerProcedures() {
        val procedureService = (graphDb as GraphDatabaseAPI).dependencyResolver.resolveDependency(
                Procedures::class.java, DependencyResolver.SelectionStrategy.FIRST
        )
        val apocProcedures = listOf(apoc.create.Create::class.java, apoc.periodic.Periodic::class.java)
        apocProcedures.forEach {
            procedureService.registerProcedure(it)
        }
    }

    /**
     * Registers a shutdown hook for the Neo4j instance so that it
     * shuts down nicely when the VM exits (even if you "Ctrl-C" the
     * running application).
     */
    private fun registerShutdownHook() {
        Runtime.getRuntime().addShutdownHook(Thread {
            graphDb.shutdown()
            databaseDirectory.deleteRecursively()
        })
    }
}