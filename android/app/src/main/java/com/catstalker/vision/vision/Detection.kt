package com.catstalker.vision.vision

data class Detection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val confidence: Float,
) {
    val width get() = right - left
    val height get() = bottom - top
}

data class DetectorResult(
    val detection: Detection?,
    val inferenceMs: Double,
)

enum class BackendMode(val label: String) {
    CPU("CPU"),
    VULKAN("NCNN Vulkan"),
}

interface Detector : AutoCloseable {
    val backend: String
    fun detect(frame: android.graphics.Bitmap): DetectorResult
}
