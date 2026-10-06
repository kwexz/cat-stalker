package com.catstalker.vision.camera

import android.util.Log
import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.catstalker.vision.benchmark.InferenceMetrics
import com.catstalker.vision.vision.BackendMode
import com.catstalker.vision.vision.Detector
import com.catstalker.vision.vision.DetectorResult
import com.catstalker.vision.vision.NcnnDetector
import com.catstalker.vision.vision.YoloDetector
import com.catstalker.vision.vision.YuvToRgbConverter
import com.catstalker.vision.session.SessionRecorder

class FrameAnalyzer(
    private val context: android.content.Context,
    private val onResult: (DetectorResult?, Int, Int, Int, String, String?) -> Unit,
) : ImageAnalysis.Analyzer, AutoCloseable {
    private val converter = YuvToRgbConverter()
    private val metrics = InferenceMetrics()
    @Volatile private var requestedMode = BackendMode.CPU
    private var activeMode: BackendMode? = null
    private var detector: Detector? = null
    @Volatile private var backendUnavailable: String? = null
    @Volatile private var nativeYuvEnabled = false
    private var lastLogNs = 0L
    @Volatile private var captureRequested = false
    @Volatile private var latestFrame: Bitmap? = null
    @Volatile var sessionRecorder: SessionRecorder? = null

    fun setBackend(mode: BackendMode) {
        metrics.reset()
        nativeYuvEnabled = mode == BackendMode.VULKAN
        requestedMode = if (mode == BackendMode.VULKAN) {
            backendUnavailable = null
            BackendMode.VULKAN
        } else {
            backendUnavailable = null
            BackendMode.CPU
        }
    }

    fun setNativeYuvEnabled(enabled: Boolean) {
        nativeYuvEnabled = enabled
        metrics.reset()
    }

    fun requestCapture() {
        captureRequested = true
    }

    override fun analyze(image: ImageProxy) {
        try {
            metrics.recordAnalysis()
            ensureDetector()
            val conversionStartedNs = System.nanoTime()
            val nativeNcnn = nativeYuvEnabled && detector is NcnnDetector
            val bitmap = if (nativeNcnn && !captureRequested) null else converter.toBitmap(image)
            val result = if (nativeNcnn) (detector as NcnnDetector).detectNativeYuv(image) else detector?.detect(requireNotNull(bitmap))
            if (captureRequested && bitmap != null) {
                latestFrame = bitmap.copy(Bitmap.Config.ARGB_8888, false)
                captureRequested = false
            }
            val conversionMs = if (nativeNcnn) 0.0 else (System.nanoTime() - conversionStartedNs) / 1_000_000.0 - (result?.inferenceMs ?: 0.0)
            if (result != null) metrics.recordInference()
            val snapshot = metrics.snapshot()
            val backend = (detector?.backend ?: "Unavailable") + if (nativeNcnn) " native YUV" else " bitmap"
            val rates = "analysis %.1f FPS | inference %.1f FPS".format(snapshot.analysisFps, snapshot.inferenceFps)
            sessionRecorder?.let { recorder ->
                if (recorder.enabled) {
                    val (battery, diskMb) = recorder.extrasProvider?.invoke()
                        ?: (SessionRecorder.BATTERY_UNKNOWN to 0L)
                    recorder.offer(
                        result?.detection, image.width, image.height,
                        image.imageInfo.rotationDegrees, backend,
                        result?.inferenceMs, snapshot.analysisFps, snapshot.inferenceFps,
                        battery, diskMb,
                    )
                    if (recorder.wantsFrame()) {
                        val frameBitmap = bitmap ?: runCatching { converter.toBitmap(image) }.getOrNull()
                        if (frameBitmap != null) recorder.offerFrame(frameBitmap, result?.detection?.confidence)
                    }
                }
            }
            logResult(result, image, rates, backend, conversionMs)
            onResult(result, image.width, image.height, image.imageInfo.rotationDegrees, backend, listOfNotNull(rates, backendUnavailable).joinToString("\n"))
        } catch (error: Exception) {
            Log.e(TAG, "Frame analysis failed", error)
            onResult(null, 0, 0, 0, "Unavailable", error.message ?: error.javaClass.simpleName)
        } finally {
            image.close()
        }
    }

    private fun ensureDetector() {
        if (activeMode == requestedMode && detector != null) return
        val previous = detector
        detector = null
        activeMode = null
        previous?.close()
        Log.i(TAG, "Creating ${requestedMode.label} detector")
        try {
            detector = if (requestedMode == BackendMode.VULKAN) NcnnDetector(context) else YoloDetector(context, BackendMode.CPU)
            activeMode = requestedMode
        } catch (error: Exception) {
            if (requestedMode != BackendMode.VULKAN) throw error
            Log.e(TAG, "NCNN Vulkan unavailable; falling back to TFLite CPU", error)
            backendUnavailable = "NCNN Vulkan unavailable; CPU fallback"
            requestedMode = BackendMode.CPU
            detector = YoloDetector(context, BackendMode.CPU)
            activeMode = BackendMode.CPU
        }
    }

    private fun logResult(result: DetectorResult?, image: ImageProxy, rates: String, backend: String, conversionMs: Double) {
        val now = System.nanoTime()
        if (now - lastLogNs < LOG_INTERVAL_NS) return
        lastLogNs = now
        val box = result?.detection?.let {
            "cat conf=%.2f box=[%.0f,%.0f,%.0f,%.0f]".format(it.confidence, it.left, it.top, it.right, it.bottom)
        } ?: "no cat"
        val inference = result?.inferenceMs?.let { "%.1f".format(it) } ?: "-"
        Log.i(TAG, "$box source=${image.width}x${image.height} rotation=${image.imageInfo.rotationDegrees} convert=${"%.1f".format(conversionMs)}ms inference=${inference}ms $rates backend=$backend")
    }


    override fun close() {
        detector?.close()
        detector = null
    }

    fun takeCapturedFrame(): Bitmap? = latestFrame.also { latestFrame = null }

    private companion object {
        const val TAG = "CatStalkerVision"
        const val LOG_INTERVAL_NS = 1_000_000_000L
    }
}
