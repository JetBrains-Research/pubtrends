plugins {
    java
    kotlin("jvm")
    id("com.github.johnrengelman.shadow") version "5.2.0"
}

repositories {
    mavenCentral()
    jcenter()
    maven("http://maven.icm.edu.pl/artifactory/repo/")
    maven("https://dl.bintray.com/rookies/maven" )
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    testCompile("junit", "junit", "4.12")
    implementation("pl.edu.icm.cermine:cermine-impl:1.12")
    implementation("org.apache.pdfbox:pdfbox:2.0.19")
    implementation("com.github.kittinunf.fuel:fuel:2.2.1")
    compile("org.grobid:grobid-core:0.5.6")
    compile("org.grobid:grobid-trainer:0.5.6")
    implementation("org.allenai:science-parse_2.11:2.0.3")
    implementation("org.apache.logging.log4j:log4j-api:2.13.1")
    implementation("org.apache.logging.log4j:log4j-core:2.13.1")
    implementation("org.apache.logging.log4j:log4j-api-kotlin:1.0.0")
    implementation("org.neo4j.driver:neo4j-java-driver:4.0.0")
    implementation("com.beust:klaxon:5.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.5")
    implementation("net.sf.jopt-simple:jopt-simple:6.0-alpha-3")

    implementation(kotlin("stdlib-jdk8"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.6.2")
    testImplementation("io.mockk:mockk:1.9.3")

    compile(project(":validation_db"))
}

configurations.all {
    exclude(group = "ch.qos.logback", module = "logback-classic")
    exclude(group = "org.slf4j", module = "slf4j-jdk14")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
}

tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }

    named<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowJar") {
        archiveBaseName.set("collector")
        mergeServiceFiles()
        manifest {
            attributes(mapOf("Main-Class" to "ArxivCollectorMainKt"))
        }
        isZip64 = true
    }
}

tasks {
    build {
        dependsOn("shadowJar")
    }
}