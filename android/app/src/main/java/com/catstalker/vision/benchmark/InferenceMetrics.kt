package com.catstalker.vision.benchmark

class InferenceMetrics {
    private var lastAnalysisNs = 0L
    private var lastInferenceNs = 0L
    private var analysisFps = 0.0
    private var inferenceFps = 0.0

    @Synchronized
    fun recordAnalysis(nowNs: Long = System.nanoTime()) {
        if (lastAnalysisNs != 0L) analysisFps = rate(analysisFps, nowNs - lastAnalysisNs)
        lastAnalysisNs = nowNs
    }

    @Synchronized
    fun recordInference(nowNs: Long = System.nanoTime()) {
        if (lastInferenceNs != 0L) inferenceFps = rate(inferenceFps, nowNs - lastInferenceNs)
        lastInferenceNs = nowNs
    }

    @Synchronized
    fun snapshot() = MetricsSnapshot(analysisFps, inferenceFps)

    @Synchronized
    fun reset() {
        lastAnalysisNs = 0L
        lastInferenceNs = 0L
        analysisFps = 0.0
        inferenceFps = 0.0
    }

    private fun rate(previous: Double, elapsedNs: Long): Double {
        if (elapsedNs <= 0) return previous
        val instantaneous = 1_000_000_000.0 / elapsedNs
        return if (previous == 0.0) instantaneous else previous * 0.8 + instantaneous * 0.2
    }
}

data class MetricsSnapshot(val analysisFps: Double, val inferenceFps: Double)
