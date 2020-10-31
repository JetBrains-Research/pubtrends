package org.jetbrains.bio.pubtrends.validation

import org.jetbrains.bio.pubtrends.data.Reference
import kotlinx.coroutines.*
import java.lang.Exception
import java.lang.Thread.sleep

interface Validator {
    fun validate(refList : List<Reference>) {
        val failedRefs = mutableListOf<Reference>()
        runBlocking {
            for (ref in refList) {
                launch {
                    try {
                        validate(ref)
                    } catch (e: Exception) {
                        failedRefs.add(ref)
                    }
                }
            }
        }

        //try failed requests again
        if (failedRefs.size != 0) {
            sleep(2000)
            var failed = 0
            runBlocking {
                for (ref in failedRefs) {
                    launch {
                        try {
                            validate(ref)
                        } catch (e: Exception) {
                            failed++
                        }
                    }
                }
            }
            if (failed != 0) {
                throw ValidatorException("Validation failed twice for some references")
            }
        }
    }

    fun validate(ref : Reference)

    class ValidatorException(override val message: String = "") : Exception(message)
}