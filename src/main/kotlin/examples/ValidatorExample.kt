package examples

// TODO(kapralov): local validator is an external dependency, not yet sure whether it is needed

//import org.jetbrains.bio.pubtrends.data.Reference
//import org.jetbrains.bio.pubtrends.validation.LocalValidator
//import kotlinx.coroutines.*
//import kotlin.system.measureTimeMillis
//
//fun main() = runBlocking {
//    val validator = LocalValidator
//    println(measureTimeMillis {
//        val refs = mutableListOf(
//            Reference("V.A. Khoze, A.D. Martin and M.G. Ryskin, Eur. Phys. J. C23 (2002) 311, hep-ph/0111078.", true)
//        )
//        refs.forEach {println(it)}
//        val job = launch {
//            validator.validate(refs)
//        }
//        job.join()
//        println(refs)
//    })
//}