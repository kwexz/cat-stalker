package com.catstalker.vision.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraController(private val context: Context) : AutoCloseable {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null
    private var cameraControl: CameraControl? = null
    private var meteringFactory: androidx.camera.core.MeteringPointFactory? = null

    fun start(owner: LifecycleOwner, previewView: PreviewView, analyzer: ImageAnalysis.Analyzer) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            provider = future.get().also { cameraProvider ->
                val preview = androidx.camera.core.Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(480, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build().also { it.setAnalyzer(executor, analyzer) }
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                cameraControl = camera.cameraControl
                meteringFactory = previewView.meteringPointFactory
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Meter auto-exposure on the tracked target (view coordinates).
     * Backlit cats otherwise expose as black silhouettes because the
     * full-frame meter follows the bright window. Best-effort: silently
     * ignored on devices without per-point AE.
     */
    fun focusOn(x: Float, y: Float) {
        val control = cameraControl ?: return
        val factory = meteringFactory ?: return
        runCatching {
            val point = factory.createPoint(x, y)
            val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(3, TimeUnit.SECONDS)
                .build()
            control.startFocusAndMetering(action)
        }.onFailure { Log.w(TAG, "metering failed", it) }
    }

    fun resetFocus() {
        runCatching { cameraControl?.cancelFocusAndMetering() }
    }

    override fun close() {
        provider?.unbindAll()
        executor.shutdownNow()
    }

    private companion object {
        const val TAG = "CatStalkerVision"
    }
}
