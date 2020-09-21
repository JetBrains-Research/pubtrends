package org.jetbrains.bio.pubtrends.db

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.exposed.sql.*
import org.jetbrains.exposed.sql.statements.BatchInsertStatement
import org.jetbrains.exposed.sql.transactions.TransactionManager
import org.postgresql.util.PGobject
import java.sql.PreparedStatement

internal const val PUBLICATION_MAX_TITLE_LENGTH = 1023
internal val jsonMapper = ObjectMapper()


/**
 * Adopted from: https://gist.github.com/quangIO/a623b5caa53c703e252d858f7a806919
 *
 * Exposed does not support JSON by default (see https://github.com/JetBrains/Exposed/issues/127)
 */

fun <T : Any> Table.jsonb(name: String, klass: Class<T>, jsonMapper: ObjectMapper): Column<T> =
        registerColumn(name, Json(klass, jsonMapper))


private class Json<out T : Any>(private val klass: Class<T>, private val jsonMapper: ObjectMapper) : ColumnType() {
    override fun sqlType() = "jsonb"

    override fun setParameter(stmt: PreparedStatement, index: Int, value: Any?) {
        val obj = PGobject()
        obj.type = "jsonb"
        obj.value = value as String
        stmt.setObject(index, obj)
    }

    override fun valueFromDB(value: Any): Any {
        value as PGobject
        return try {
            jsonMapper.readValue(value.value, klass)
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Can't parse JSON: $value")
        }
    }


    override fun notNullValueToDB(value: Any): Any = jsonMapper.writeValueAsString(value)
    override fun nonNullValueToString(value: Any): String = "'${jsonMapper.writeValueAsString(value)}'"
}

class PGEnum<T : Enum<T>>(enumTypeName: String, enumValue: T?) : PGobject() {
    init {
        value = enumValue?.name
        type = enumTypeName
    }
}

class BatchInsertUpdateOnDuplicate(
        table: Table,
        private val column: Column<*>,
        private val onDupUpdate: List<Column<*>>
) : BatchInsertStatement(table, false) {
    override fun prepareSQL(transaction: Transaction): String {
        val onUpdateSQL = if (onDupUpdate.isNotEmpty()) {
            " ON CONFLICT (${column.name}) DO UPDATE SET ${
                onDupUpdate.joinToString {
                    "${transaction.identity(it)} = Excluded.${transaction.identity(it)}"
                }
            }"
        } else ""
        return super.prepareSQL(transaction) + onUpdateSQL
    }
}

fun <T : Table, E> T.batchInsertOnDuplicateKeyUpdate(
        data: List<E>,
        column: Column<*>,
        onDupUpdateColumns: List<Column<*>>, body: T.(BatchInsertUpdateOnDuplicate, E) -> Unit
): List<Int> {
    return data.takeIf { it.isNotEmpty() }?.let {
        val insert = BatchInsertUpdateOnDuplicate(this, column, onDupUpdateColumns)
        data.forEach {
            insert.addBatch()
            body(insert, it)
        }
        TransactionManager.current().exec(insert)
        columns.firstOrNull { it.columnType.isAutoInc }?.let { idCol ->
            insert.generatedKey?.mapNotNull {
                val value = it[idCol]
                when (value) {
                    is Long -> value.toInt()
                    is Int -> value
                    null -> null
                    else -> error(
                            "can't find primary key of type Int or Long; " +
                                    "map['$idCol']='$value' (where map='$it')"
                    )
                }
            }
        }
    }.orEmpty()
}