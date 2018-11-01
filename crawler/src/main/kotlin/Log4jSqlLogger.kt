import org.apache.logging.log4j.LogManager
import org.jetbrains.exposed.sql.SqlLogger
import org.jetbrains.exposed.sql.Transaction
import org.jetbrains.exposed.sql.statements.StatementContext
import org.jetbrains.exposed.sql.statements.expandArgs

object Log4jSqlLogger : SqlLogger {
    private val logger = LogManager.getLogger(Log4jSqlLogger::class)

    override fun log(context: StatementContext, transaction: Transaction) {
        logger.debug("SQL: ${context.expandArgs(transaction)}")
    }
}