package com.catstalker.vision.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import com.catstalker.vision.vision.Detection

class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 183, 45)
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private var detection: Detection? = null
    private var detectionUpdatedNs = 0L
    @Volatile private var lastCenter: android.graphics.PointF? = null

    /** Center of the last drawn box, in this view's coordinates. */
    fun lastCenter(): android.graphics.PointF? = lastCenter
    private var imageWidth = 0
    private var imageHeight = 0
    private var previewRotation = 0
    private var lastLogNs = 0L

    fun update(detection: Detection?, imageWidth: Int, imageHeight: Int, rotation: Int) {
        if (detection != null) {
            this.detection = detection
            detectionUpdatedNs = System.nanoTime()
        } else if (System.nanoTime() - detectionUpdatedNs > DETECTION_HOLD_NS) {
            this.detection = null
        }
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        this.previewRotation = rotation
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        val value = detection ?: run { lastCenter = null; return }
        if (imageWidth == 0 || imageHeight == 0) return
        val rotatedWidth = if (previewRotation % 180 == 0) imageWidth else imageHeight
        val rotatedHeight = if (previewRotation % 180 == 0) imageHeight else imageWidth
        val scale = maxOf(width.toFloat() / rotatedWidth, height.toFloat() / rotatedHeight)
        val offsetX = (width - rotatedWidth * scale) / 2f
        val offsetY = (height - rotatedHeight * scale) / 2f
        val source = when (previewRotation) {
            90 -> RectF(imageHeight - value.bottom, value.left, imageHeight - value.top, value.right)
            180 -> RectF(imageWidth - value.right, imageHeight - value.bottom, imageWidth - value.left, imageHeight - value.top)
            270 -> RectF(value.top, imageWidth - value.right, value.bottom, imageWidth - value.left)
            else -> RectF(value.left, value.top, value.right, value.bottom)
        }
        val rect = RectF(
            offsetX + source.left * scale,
            offsetY + source.top * scale,
            offsetX + source.right * scale,
            offsetY + source.bottom * scale,
        )
        canvas.drawRect(rect, boxPaint)
        lastCenter = android.graphics.PointF(rect.centerX(), rect.centerY())
        canvas.drawText("cat %.2f".format(value.confidence), rect.left, maxOf(labelPaint.textSize, rect.top - 10f), labelPaint)
        logDraw("rect=[%.0f,%.0f,%.0f,%.0f] source=${imageWidth}x$imageHeight rotation=$previewRotation".format(rect.left, rect.top, rect.right, rect.bottom))
    }

    private fun logDraw(message: String) {
        val now = System.nanoTime()
        if (now - lastLogNs < LOG_INTERVAL_NS) return
        lastLogNs = now
        Log.i(TAG, "overlay $message")
    }

    private companion object {
        const val TAG = "CatStalkerVision"
        const val LOG_INTERVAL_NS = 1_000_000_000L
        const val DETECTION_HOLD_NS = 2_000_000_000L
    }
}
