package com.preprint.server.core.ref

import com.preprint.server.core.data.Reference
import com.preprint.server.core.validation.Validator
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.apache.logging.log4j.kotlin.logger
import java.lang.Thread.sleep

interface ReferenceExtractor {
    fun getReferences(pdf : ByteArray, validators : List<Validator> = listOf()) : List<Reference> {
        val refs = extractUnverifiedReferences(pdf)
        logger().info("Begin validation of ${refs.size} references")
        var attemptsDone = 0
        while (true) {
            var stopValidation = true
            try {
                validators.forEach { validator ->
                    validator.validate(refs)
                }
            } catch (e: Validator.ValidatorException) {
                stopValidation = false
                attemptsDone += 1
                logger().error(e.message)
                sleep(2000)
                if (attemptsDone >= 3) {
                    stopValidation = true
                }
            }
            if (stopValidation) {
                break
            }
        }
        if (validators.isNotEmpty()) {
            logger().info("Validated ${refs.count { it.validated }} out of ${refs.size}")
        }
        return refs
    }
    fun extractUnverifiedReferences(pdf : ByteArray) : List<Reference>
}