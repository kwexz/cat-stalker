package com.catstalker.vision.session

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.catstalker.vision.vision.Detection
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.ArrayDeque
import java.util.UUID

/** One telemetry row, mirroring session/protocol.py PHONE_CSV_HEADER. */
data class TelemetryRow(
    val ts: Double,
    val state: String,
    val confidence: Float?,
    val offsetX: Float?,
    val areaRatio: Float?,
    val inferenceMs: Double?,
    val analysisFps: Double,
    val inferenceFps: Double,
    val backend: String,
    val batteryPct: Int,
    val diskFreeMb: Long,
)

/**
 * Thin buffer: telemetry ring in memory, event JPEGs in a size-capped
 * disk queue. Everything is deleted after the PC collector acks it.
 * Phone keeps no large history during training/tuning.
 */
class SessionRecorder(context: Context) {
    @Volatile var enabled: Boolean = false

    /** Supplies (batteryPct, diskFreeMb); set by the uploader (has Context). */
    @Volatile var extrasProvider: (() -> Pair<Int, Long>)? = null

    private val queueDir = File(context.cacheDir, "session_queue").apply { mkdirs() }
    private val rows = ArrayDeque<TelemetryRow>()
    private val lock = Any()

    private var wasDetected = false
    private var lastPeriodicNs = 0L
    private var lastLowConfNs = 0L

    var sessionId: String = ""
        private set

    fun start(): String {
        synchronized(lock) {
            rows.clear()
            sessionId = UUID.randomUUID().toString().take(8)
            enabled = true
            return sessionId
        }
    }

    fun stop() {
        enabled = false
    }

    fun offer(
        detection: Detection?,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int,
        backend: String,
        inferenceMs: Double?,
        analysisFps: Double,
        inferenceFps: Double,
        batteryPct: Int,
        diskFreeMb: Long,
    ) {
        if (!enabled || imageWidth == 0 || imageHeight == 0) return
        val nowNs = System.nanoTime()
        val detected = detection != null
        val row = TelemetryRow(
            ts = nowNs / 1_000_000_000.0,
            state = if (detected) "DETECTED" else "SEARCHING",
            confidence = detection?.confidence,
            offsetX = detection?.let { azimuthOffset(it, imageWidth, imageHeight, rotationDegrees) },
            areaRatio = detection?.let { it.width * it.height / (imageWidth.toFloat() * imageHeight) },
            inferenceMs = inferenceMs,
            analysisFps = analysisFps,
            inferenceFps = inferenceFps,
            backend = backend,
            batteryPct = batteryPct,
            diskFreeMb = diskFreeMb,
        )
        synchronized(lock) {
            if (rows.size >= MAX_ROWS) rows.removeFirst()
            rows.addLast(row)
        }
        // Frame policy is evaluated by FrameAnalyzer via wantsFrame();
        // transition/low-conf bookkeeping lives here.
        if (detected != wasDetected) {
            wasDetected = detected
            frameDue = true
        } else if (detected && nowNs - lastPeriodicNs > PERIODIC_NS) {
            lastPeriodicNs = nowNs
            frameDue = true
        } else if (detected && (detection?.confidence ?: 1f) < 0.35f &&
            nowNs - lastLowConfNs > LOW_CONF_NS
        ) {
            lastLowConfNs = nowNs
            frameDue = true
        }
    }

    @Volatile private var frameDue = false

    /** Called by the analyzer thread; true at most ~1/3s. */
    fun wantsFrame(): Boolean {
        if (!enabled || !frameDue) return false
        frameDue = false
        return true
    }

    fun offerFrame(bitmap: Bitmap, confidence: Float?, kind: String = "event") {
        if (!enabled) return
        try {
            evictIfNeeded()
            val ts = System.currentTimeMillis() / 1000.0
            val conf = confidence?.let { "%.2f".format(it) } ?: "na"
            val file = File(queueDir, "${ts}_${kind}_conf${conf}.jpg")
            file.outputStream().use { out ->
                ByteArrayOutputStream().use { buf ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 70, buf)
                    out.write(buf.toByteArray())
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "frame queue write failed", e)
        }
    }

    fun drainRows(max: Int): List<TelemetryRow> {
        synchronized(lock) {
            val out = mutableListOf<TelemetryRow>()
            repeat(minOf(max, rows.size)) { out.add(rows.removeFirst()) }
            return out
        }
    }

    fun pendingFrames(): List<File> {
        val files = queueDir.listFiles()?.sortedBy { it.lastModified() } ?: return emptyList()
        return files.filter { it.name.endsWith(".jpg") }
    }

    fun pendingRowCount(): Int = synchronized(lock) { rows.size }

    private fun evictIfNeeded() {
        val files = queueDir.listFiles()?.sortedBy { it.lastModified() } ?: return
        var total = files.sumOf { it.length() }
        for (f in files) {
            if (total <= FRAME_QUEUE_CAP_BYTES) break
            total -= f.length()
            f.delete()
        }
    }

    companion object {
        /**
         * Horizontal (azimuth) offset of the box center in [-1, 1],
         * + = cat right of frame center. Boxes live in sensor-buffer
         * coords, so for a rotated buffer the azimuth axis is the
         * buffer Y, not X. Mapping mirrors DetectionOverlay::onDraw:
         * screenX(90) = H - bufferY, screenX(270) = bufferY,
         * screenX(180) = W - bufferX. The old code always used
         * buffer-X, i.e. steered by the cat's HEIGHT: constant error,
         * endless spin, never centering.
         */
        fun azimuthOffset(
            d: Detection,
            imageWidth: Int,
            imageHeight: Int,
            rotationDegrees: Int,
        ): Float {
            val cy = (d.top + d.bottom) / 2f / imageHeight.toFloat()
            val cx = (d.left + d.right) / 2f / imageWidth.toFloat()
            return when (((rotationDegrees % 360) + 360) % 360) {
                90 -> -((cy - 0.5f) * 2f)
                270 -> ((cy - 0.5f) * 2f)
                180 -> -((cx - 0.5f) * 2f)
                else -> ((cx - 0.5f) * 2f)
            }
        }
        const val BATTERY_UNKNOWN = -1
        private const val TAG = "CatStalkerSession"
        private const val MAX_ROWS = 2000
        private const val PERIODIC_NS = 5_000_000_000L
        private const val LOW_CONF_NS = 5_000_000_000L
        private const val FRAME_QUEUE_CAP_BYTES = 100L * 1024 * 1024
    }
}
