package com.catstalker.vision

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.MediaStore
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.catstalker.vision.session.SessionConfig
import com.catstalker.vision.session.SessionDiscovery
import com.catstalker.vision.session.SessionRecorder
import com.catstalker.vision.session.SessionUploader
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.catstalker.vision.camera.CameraController
import com.catstalker.vision.camera.FrameAnalyzer
import com.catstalker.vision.ui.DetectionOverlay
import com.catstalker.vision.vision.BackendMode
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private lateinit var camera: CameraController
    private lateinit var analyzer: FrameAnalyzer
    private lateinit var preview: PreviewView
    private lateinit var overlay: DetectionOverlay
    private lateinit var status: TextView
    private lateinit var telemetry: TextView
    private lateinit var buildStamp: TextView
    private lateinit var backendButton: Button
    private lateinit var captureButton: Button
    private lateinit var sessionHost: EditText
    private lateinit var sessionButton: Button
    private lateinit var sessionStatus: TextView
    private lateinit var sessionRecorder: SessionRecorder
    private lateinit var sessionUploader: SessionUploader
    private lateinit var sessionConfig: SessionConfig
    // Autonomous default: boot straight into the operating backend.
    private var mode = BackendMode.VULKAN
    private var autoSessionDone = false
    private var lastMeteringNs = 0L
    private var meteringActive = false
    private var nativeYuvEnabled = false
    private var lastUiDetectedNs = 0L
    private lateinit var powerManager: PowerManager
    private var sustainedPerformanceSupported = false

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text = "CAMERA PERMISSION REQUIRED"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Autonomous sessions: never let the screen sleep while visible.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        status = findViewById(R.id.status)
        telemetry = findViewById(R.id.telemetry)
        buildStamp = findViewById(R.id.buildStamp)
        backendButton = findViewById(R.id.backendButton)
        captureButton = findViewById(R.id.captureButton)
        sessionHost = findViewById(R.id.sessionHost)
        sessionButton = findViewById(R.id.sessionButton)
        sessionStatus = findViewById(R.id.sessionStatus)
        sessionConfig = SessionConfig.load(this)
        sessionHost.setText(sessionConfig.host)
        sessionRecorder = SessionRecorder(applicationContext)
        sessionUploader = SessionUploader(applicationContext, sessionRecorder) { event ->
            runOnUiThread { sessionStatus.text = event }
        }
        sessionRecorder.extrasProvider = {
            sessionUploader.batteryPct() to sessionUploader.diskFreeMb()
        }
        powerManager = getSystemService(PowerManager::class.java)
        sustainedPerformanceSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && powerManager.isSustainedPerformanceModeSupported
        preview.scaleType = PreviewView.ScaleType.FILL_CENTER
        preview.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        buildStamp.text = "build ${BuildConfig.BUILD_STAMP} p30-finetune-v5-dino"
        backendButton.text = "Switch to TFLite CPU"
        camera = CameraController(this)
        analyzer = FrameAnalyzer(applicationContext, ::renderResult)
        analyzer.sessionRecorder = sessionRecorder
        sessionButton.setOnClickListener {
            if (sessionUploader.isRunning()) {
                sessionUploader.stop()
                sessionButton.text = "Start Session Upload"
            } else {
                sessionConfig.host = sessionHost.text.toString().trim()
                SessionConfig.save(this, sessionConfig)
                sessionUploader.reloadConfig()
                sessionUploader.start()
                sessionButton.text = "Stop Session Upload"
            }
        }
        backendButton.setOnClickListener {
            mode = if (mode == BackendMode.CPU) BackendMode.VULKAN else BackendMode.CPU
            nativeYuvEnabled = mode == BackendMode.VULKAN
            setSustainedPerformance(mode == BackendMode.VULKAN)
            analyzer.setBackend(mode)
            backendButton.text = if (mode == BackendMode.CPU) "Switch to P30-tuned v5 NCNN Vulkan 384" else "Switch to TFLite CPU"
            status.text = if (mode == BackendMode.VULKAN) "SWITCHING TO NCNN VULKAN + NATIVE BILINEAR" else "SWITCHING TO TFLITE CPU"
        }
        backendButton.setOnLongClickListener {
            nativeYuvEnabled = !nativeYuvEnabled
            analyzer.setNativeYuvEnabled(nativeYuvEnabled)
            status.text = if (nativeYuvEnabled) "NATIVE YUV TEST" else "BITMAP PREPROCESSING"
            backendButton.text = if (nativeYuvEnabled) "Native YUV active (long-press to restore)" else "P30-tuned v5 bitmap active"
            true
        }
        captureButton.setOnClickListener {
            analyzer.requestCapture()
            captureButton.text = "Capturing..."
            captureButton.postDelayed({ saveHardNegative() }, 300)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        analyzer.setBackend(mode)
        camera.start(this, preview, analyzer)
        autoStartSessionOnce()
    }

    /**
     * Zero-touch steady state: discover the PC collector and start
     * uploading without any taps. Manual Start/Stop button stays.
     */
    private fun autoStartSessionOnce() {
        if (autoSessionDone) return
        autoSessionDone = true
        val activity = this@MainActivity
        val worker = object : Thread("SessionAutoStart") {
            override fun run() {
                var ready = sessionConfig.isConfigured()
                if (!ready) {
                    activity.runOnUiThread { sessionStatus.text = "SESSION: discovering PC..." }
                    val found = SessionDiscovery.discover(activity)
                    if (found != null) {
                        sessionConfig.host = found.host
                        sessionConfig.port = found.port
                        SessionConfig.save(activity, sessionConfig)
                        activity.runOnUiThread { sessionHost.setText(sessionConfig.host) }
                        ready = true
                    } else {
                        activity.runOnUiThread { sessionStatus.text = "SESSION: PC not found, set host manually" }
                    }
                }
                if (!ready) return
                sessionUploader.reloadConfig()
                sessionUploader.start()
                activity.runOnUiThread { sessionButton.text = "Stop Session Upload" }
            }
        }
        worker.isDaemon = true
        worker.start()
    }

    private fun renderResult(result: com.catstalker.vision.vision.DetectorResult?, width: Int, height: Int, rotation: Int, backend: String, detail: String?) {
        runOnUiThread {
            val detection = result?.detection
            if (detection != null) lastUiDetectedNs = System.nanoTime()
            val isDetected = System.nanoTime() - lastUiDetectedNs < 2_000_000_000L
            overlay.update(detection, width, height, rotation)
            status.text = if (isDetected) "DETECTED" else "SEARCHING"
            // Meter exposure on the tracked cat: backlit targets expose
            // as silhouettes under full-frame metering. Throttled; the
            // action auto-cancels, reset explicitly when lost.
            val nowNs = System.nanoTime()
            if (isDetected) {
                val center = overlay.lastCenter()
                if (center != null && nowNs - lastMeteringNs > 2_500_000_000L) {
                    lastMeteringNs = nowNs
                    meteringActive = true
                    camera.focusOn(center.x, center.y)
                }
            } else if (meteringActive) {
                meteringActive = false
                camera.resetFocus()
            }
            val values = if (detection == null) {
                "conf: -\noffset_x: -\narea: -\nbbox: -"
            } else {
                val offset = ((detection.left + detection.right) / 2f / width - 0.5f) * 2f
                val area = detection.width * detection.height / (width.toFloat() * height)
                "conf: %.2f\noffset_x: %+.2f\narea: %.1f%%\nbbox: [%.0f %.0f %.0f %.0f]".format(
                    Locale.US,
                    detection.confidence,
                    offset,
                    area * 100f,
                    detection.left,
                    detection.top,
                    detection.right,
                    detection.bottom,
                )
            }
            telemetry.text = "$values\ninference: ${result?.inferenceMs?.let { "%.1f ms".format(Locale.US, it) } ?: "-"}\n$detail\nbackend: $backend"
        }
    }

    override fun onDestroy() {
        setSustainedPerformance(false)
        analyzer.close()
        camera.close()
        super.onDestroy()
    }

    private fun setSustainedPerformance(enabled: Boolean) {
        if (!sustainedPerformanceSupported) return
        window.setSustainedPerformanceMode(enabled)
        Log.i("CatStalkerVision", "Sustained performance mode=${enabled}")
    }

    private fun saveHardNegative() {
        val frame = analyzer.takeCapturedFrame()
        if (frame == null) {
            captureButton.text = "Save Hard Negative"
            return
        }
        val values = android.content.ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "CAT_STALKER_NEG_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CatStalker/HardNegatives")
        }
        contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)?.let { uri ->
            contentResolver.openOutputStream(uri)?.use { stream -> frame.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, stream) }
            status.text = "HARD NEGATIVE SAVED"
        }
        captureButton.text = "Save Hard Negative"
    }
}
