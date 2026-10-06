package com.catstalker.vision.vision

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

class YuvToRgbConverter {
    fun toBitmap(image: ImageProxy): Bitmap {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            val yRow = y * yPlane.rowStride
            val uvRow = (y / 2) * uPlane.rowStride
            for (x in 0 until width) {
                val luma = yPlane.buffer.get(yRow + x * yPlane.pixelStride).toInt() and 0xff
                val u = (uPlane.buffer.get(uvRow + (x / 2) * uPlane.pixelStride).toInt() and 0xff) - 128
                val v = (vPlane.buffer.get((y / 2) * vPlane.rowStride + (x / 2) * vPlane.pixelStride).toInt() and 0xff) - 128
                val r = (luma + 1.402f * v).toInt().coerceIn(0, 255)
                val g = (luma - 0.344136f * u - 0.714136f * v).toInt().coerceIn(0, 255)
                val b = (luma + 1.772f * u).toInt().coerceIn(0, 255)
                pixels[y * width + x] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
