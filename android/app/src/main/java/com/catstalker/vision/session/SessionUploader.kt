package com.catstalker.vision.session

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

/**
 * Background uploader: telemetry batches every 0.5s (follow mode
 * needs fresh frames: the PC stops the robot on frames older than
 * 0.5s), oldest JPEG first, heartbeat every 5s, polls the PC stop
 * flag. Stdlib HTTP only.
 */
class SessionUploader(
    private val context: Context,
    private val recorder: SessionRecorder,
    private val onEvent: (String) -> Unit,
) {
    @Volatile private var running = false
    private var thread: Thread? = null
    private var config = SessionConfig.load(context)

    fun reloadConfig() {
        config = SessionConfig.load(context)
    }

    fun batteryPct(): Int {
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val status = context.registerReceiver(null, filter) ?: return -1
        val level = status.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = status.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        return if (level >= 0 && scale > 0) (level * 100 / scale) else -1
    }

    fun diskFreeMb(): Long = context.cacheDir.usableSpace / (1024 * 1024)

    fun start() {
        if (running) return
        if (!config.isConfigured()) {
            onEvent("SESSION: set PC host first")
            return
        }
        running = true
        val sessionId = recorder.start()
        thread = Thread({
            var lastBatchNs = 0L
            var lastBeatNs = 0L
            try {
                postJson("/api/v1/hello", JSONObject().apply {
                    put("device_id", config.deviceId)
                    put("backend", "android")
                    put("session_id", sessionId)
                })
            } catch (e: Exception) {
                Log.w(TAG, "hello failed", e)
            }
            while (running && recorder.enabled) {
                val now = System.nanoTime()
                try {
                    if (now - lastBatchNs > 500_000_000L) {
                        lastBatchNs = now
                        uploadBatch(sessionId)
                        uploadFrames(sessionId)
                    }
                    if (now - lastBeatNs > 5_000_000_000L) {
                        lastBeatNs = now
                        heartbeat(sessionId)
                        if (pollStop(sessionId)) {
                            recorder.stop()
                            onEvent("SESSION STOPPED BY PC")
                            break
                        }
                    }
                    consecutiveFailures = 0
                } catch (e: Exception) {
                    Log.w(TAG, "upload loop", e)
                    if (++consecutiveFailures >= REDISCOVER_AFTER_FAILURES) {
                        consecutiveFailures = 0
                        rediscoverCollector()
                    }
                }
                try {
                    Thread.sleep(500)
                } catch (e: InterruptedException) {
                    break
                }
            }
            running = false
        }, "SessionUploader").also { it.isDaemon = true; it.start() }
        onEvent("SESSION UPLOADING -> ${config.baseUrl()}")
    }

    fun stop() {
        running = false
        recorder.stop()
        thread?.interrupt()
        thread = null
        onEvent("SESSION STOPPED")
    }

    fun isRunning(): Boolean = running

    @Volatile private var consecutiveFailures = 0

    /**
     * If the collector died and a new session started elsewhere,
     * find it by broadcast instead of waiting for a human with usB.
     */
    private fun rediscoverCollector() {
        onEvent("SESSION: re-discovering PC...")
        val found = SessionDiscovery.discover(context) ?: return
        config.host = found.host
        config.port = found.port
        SessionConfig.save(context, config)
        onEvent("SESSION: collector at ${config.baseUrl()}")
    }

    private fun uploadBatch(sessionId: String) {
        val rows = recorder.drainRows(200)
        if (rows.isEmpty()) return
        val arr = JSONArray()
        for (r in rows) {
            arr.put(JSONObject().apply {
                put("ts", r.ts)
                put("state", r.state)
                r.confidence?.let { put("confidence", it.toDouble()) }
                r.offsetX?.let { put("offset_x", it.toDouble()) }
                r.areaRatio?.let { put("area_ratio", it.toDouble()) }
                r.inferenceMs?.let { put("inference_ms", it) }
                put("analysis_fps", r.analysisFps)
                put("inference_fps", r.inferenceFps)
                put("backend", r.backend)
                put("battery_pct", r.batteryPct)
                put("disk_free_mb", r.diskFreeMb)
            })
        }
        val payload = JSONObject().apply {
            put("session_id", sessionId)
            put("device_id", config.deviceId)
            put("rows", arr)
        }
        postJson("/api/v1/telemetry", payload)
    }

    private fun uploadFrames(sessionId: String) {
        for (file in recorder.pendingFrames().take(4)) {
            val parts = file.nameWithoutExtension.split("_")
            val conf = parts.lastOrNull()?.removePrefix("conf") ?: "na"
            val kind = if (parts.size > 1) parts[parts.size - 2] else "event"
            val url = URL("${config.baseUrl()}/api/v1/frame?session_id=$sessionId&kind=$kind&ts=${file.lastModified() / 1000.0}&conf=$conf")
            (url.openConnection() as HttpURLConnection).run {
                connectTimeout = 5000
                readTimeout = 15000
                requestMethod = "POST"
                setRequestProperty("Content-Type", "image/jpeg")
                doOutput = true
                outputStream.use { out -> file.inputStream().use { it.copyTo(out) } }
                val code = responseCode
                disconnect()
                if (code == 200) file.delete() else return
            }
        }
    }

    private fun heartbeat(sessionId: String) {
        postJson("/api/v1/hello", JSONObject().apply {
            put("device_id", config.deviceId)
            put("session_id", sessionId)
            put("battery_pct", batteryPct())
            put("disk_free_mb", diskFreeMb())
            put("queued_rows", recorder.pendingRowCount())
            put("queued_frames", recorder.pendingFrames().size)
        })
    }

    private fun pollStop(sessionId: String): Boolean {
        val url = URL("${config.baseUrl()}/api/v1/session?session_id=$sessionId")
        (url.openConnection() as HttpURLConnection).run {
            connectTimeout = 5000
            readTimeout = 5000
            requestMethod = "GET"
            return try {
                val code = responseCode
                if (code != 200) return false
                val body = inputStream.bufferedReader().readText()
                disconnect()
                JSONObject(body).optBoolean("stop", false)
            } finally {
                disconnect()
            }
        }
    }

    private fun postJson(path: String, payload: JSONObject) {
        val url = URL(config.baseUrl() + path)
        (url.openConnection() as HttpURLConnection).run {
            connectTimeout = 5000
            readTimeout = 10000
            requestMethod = "POST"
            setRequestProperty("Content-Type", "application/json")
            doOutput = true
            OutputStreamWriter(outputStream, Charsets.UTF_8).use { it.write(payload.toString()) }
            val code = responseCode
            disconnect()
            if (code != 200) throw java.io.IOException("HTTP $code for $path")
        }
    }

    companion object {
        private const val TAG = "CatStalkerSession"
        /** ~30 s of failed cycles triggers broadcast re-discovery. */
        private const val REDISCOVER_AFTER_FAILURES = 12
    }
}
