package org.jetbrains.bio.pubtrends.db

import com.fasterxml.jackson.databind.ObjectMapper
import org.jetbrains.exposed.v1.core.Column
import org.jetbrains.exposed.v1.core.Table
import org.jetbrains.exposed.v1.json.jsonb
import org.postgresql.util.PGobject

internal const val PUBLICATION_MAX_TITLE_LENGTH = 1023
internal val jsonMapper = ObjectMapper()

fun <T : Any> Table.jsonb(name: String, klass: Class<T>, jsonMapper: ObjectMapper): Column<T> =
    jsonb(name, { jsonMapper.writeValueAsString(it) }, { jsonMapper.readValue(it, klass) })

class PGEnum<T : Enum<T>>(enumTypeName: String, enumValue: T?) : PGobject() {
    init {
        value = enumValue?.name
        type = enumTypeName
    }
}