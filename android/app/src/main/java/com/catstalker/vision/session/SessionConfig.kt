package com.catstalker.vision.session

import android.content.Context
import android.content.SharedPreferences

/** Autonomous-session endpoint config. Set once at deploy, then hands-free. */
data class SessionConfig(
    var host: String = "",
    var port: Int = 8765,
    val deviceId: String = "p30",
) {
    fun baseUrl(): String = "http://$host:$port"
    fun isConfigured(): Boolean = host.isNotBlank()

    companion object {
        private const val PREFS = "cat_stalker_session"
        private const val KEY_HOST = "host"
        private const val KEY_PORT = "port"

        fun load(context: Context): SessionConfig {
            val prefs: SharedPreferences =
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            return SessionConfig(
                host = prefs.getString(KEY_HOST, "") ?: "",
                port = prefs.getInt(KEY_PORT, 8765),
            )
        }

        fun save(context: Context, config: SessionConfig) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY_HOST, config.host)
                .putInt(KEY_PORT, config.port)
                .apply()
        }
    }
}
