package com.catstalker.vision.vision

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

class NcnnDetector(context: Context) : Detector {
    override val backend: String
    private val handle: Long

    init {
        handle = create(context.assets)
        require(handle != 0L) { "NCNN Vulkan initialization failed; see CatStalkerVision logcat" }
        backend = "NCNN Vulkan 384 FP16 P30-tuned v5"
    }

    override fun detect(frame: Bitmap): DetectorResult {
        val startedNs = System.nanoTime()
        val values = detect(handle, frame)
        val detection = if (values == null || values.size != 5) null else Detection(values[0], values[1], values[2], values[3], values[4])
        return DetectorResult(detection, (System.nanoTime() - startedNs) / 1_000_000.0)
    }

    fun detectNativeYuv(image: ImageProxy): DetectorResult {
        val startedNs = System.nanoTime()
        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]
        val values = detectYuv(
            handle,
            y.buffer, y.rowStride, y.pixelStride,
            u.buffer, u.rowStride, u.pixelStride,
            v.buffer, v.rowStride, v.pixelStride,
            image.width, image.height,
        )
        val detection = if (values == null || values.size != 5) null else Detection(values[0], values[1], values[2], values[3], values[4])
        return DetectorResult(detection, (System.nanoTime() - startedNs) / 1_000_000.0)
    }

    override fun close() {
        destroy(handle)
    }

    private external fun create(assetManager: android.content.res.AssetManager): Long
    private external fun detect(handle: Long, bitmap: Bitmap): FloatArray?
    private external fun detectYuv(
        handle: Long,
        y: java.nio.ByteBuffer, yRowStride: Int, yPixelStride: Int,
        u: java.nio.ByteBuffer, uRowStride: Int, uPixelStride: Int,
        v: java.nio.ByteBuffer, vRowStride: Int, vPixelStride: Int,
        width: Int, height: Int,
    ): FloatArray?
    private external fun destroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("cat_stalker_ncnn")
        }
    }
}
