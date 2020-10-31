package examples

import com.preprint.server.core.validation.CrossRefValidator
import com.preprint.server.core.data.Reference
import com.preprint.server.core.validation.LocalValidator
import kotlinx.coroutines.*
import kotlin.system.measureTimeMillis

fun main() = runBlocking {
    val validator = LocalValidator
    println(measureTimeMillis {
        val refs = mutableListOf(
            Reference("V.A. Khoze, A.D. Martin and M.G. Ryskin, Eur. Phys. J. C23 (2002) 311, hep-ph/0111078.", true)
        )
        refs.forEach {println(it)}
        val job = launch {
            validator.validate(refs)
        }
        job.join()
        println(refs)
    })
}