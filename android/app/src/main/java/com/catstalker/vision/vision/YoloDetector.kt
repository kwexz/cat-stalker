package com.catstalker.vision.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    context: Context,
    mode: BackendMode,
) : Detector {
    override val backend: String
    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputType: DataType
    private val outputShape: IntArray
    private val outputCount: Int
    private val outputChannels: Int
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val inputBitmap: Bitmap
    private val inputCanvas: Canvas
    private val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
    private val pixels: IntArray

    init {
        val options = Interpreter.Options().apply { setNumThreads(4) }
        require(mode == BackendMode.CPU) { "GPU benchmark is not available in this build" }
        backend = "CPU"

        interpreter = Interpreter(loadModel(context), options)
        val input = interpreter.getInputTensor(0)
        inputWidth = input.shape()[3]
        inputHeight = input.shape()[2]
        inputType = input.dataType()
        require(input.shape().contentEquals(intArrayOf(1, 3, inputHeight, inputWidth))) {
            "Expected NCHW RGB input, got ${input.shape().contentToString()}"
        }

        val output = interpreter.getOutputTensor(0)
        outputShape = output.shape()
        require(outputShape.size == 3 && outputShape[0] == 1) {
            "Expected one YOLO output tensor, got ${outputShape.contentToString()}"
        }
        outputCount = outputShape[2]
        outputChannels = outputShape[1]
        require(outputChannels >= 84) { "Expected COCO YOLO head [1, channels, boxes], got ${outputShape.contentToString()}" }
        require(output.dataType() == DataType.FLOAT32) { "Export a float32 TFLite model; got ${output.dataType()}" }

        inputBuffer = ByteBuffer.allocateDirect(input.numBytes()).order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(interpreter.getOutputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
        inputBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        inputCanvas = Canvas(inputBitmap)
        pixels = IntArray(inputWidth * inputHeight)
        Log.i(TAG, "YOLO input=${input.shape().contentToString()} output=${outputShape.contentToString()} backend=$backend")
    }

    override fun detect(frame: Bitmap): DetectorResult {
        val startedNs = System.nanoTime()
        val transform = letterbox(frame)
        inputBuffer.rewind()
        inputBuffer.putPixels()
        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        val result = parseOutput(transform)
        return DetectorResult(result, (System.nanoTime() - startedNs) / 1_000_000.0)
    }

    private fun letterbox(source: Bitmap): LetterboxTransform {
        val scale = min(inputWidth.toFloat() / source.width, inputHeight.toFloat() / source.height)
        val resizedWidth = (source.width * scale).toInt()
        val resizedHeight = (source.height * scale).toInt()
        val padX = (inputWidth - resizedWidth) / 2
        val padY = (inputHeight - resizedHeight) / 2
        inputCanvas.drawColor(Color.rgb(114, 114, 114))
        inputCanvas.drawBitmap(source, null, RectF(padX.toFloat(), padY.toFloat(), (padX + resizedWidth).toFloat(), (padY + resizedHeight).toFloat()), paint)
        inputBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        return LetterboxTransform(scale, padX.toFloat(), padY.toFloat(), source.width, source.height)
    }

    private fun ByteBuffer.putPixels() {
        for (channel in 0..2) {
            for (pixel in pixels) {
                val value = when (channel) {
                    0 -> (pixel shr 16) and 0xff
                    1 -> (pixel shr 8) and 0xff
                    else -> pixel and 0xff
                }
                when (inputType) {
                    DataType.FLOAT32 -> putFloat(value / 255f)
                    DataType.UINT8 -> put(value.toByte())
                    else -> error("Unsupported input type: $inputType")
                }
            }
        }
    }

    private fun parseOutput(transform: LetterboxTransform): Detection? {
        val values = FloatArray(outputCount * outputChannels)
        outputBuffer.asFloatBuffer().get(values)
        val candidates = ArrayList<Detection>()
        for (index in 0 until outputCount) {
            val cx = value(values, index, 0) * inputWidth
            val cy = value(values, index, 1) * inputHeight
            val width = value(values, index, 2) * inputWidth
            val height = value(values, index, 3) * inputHeight
            val confidence = value(values, index, CAT_CLASS + 4)
            if (confidence < CONFIDENCE || width <= 0f || height <= 0f) continue
            val candidate = Detection(
                left = ((cx - width / 2f - transform.padX) / transform.scale).coerceIn(0f, transform.sourceWidth.toFloat()),
                top = ((cy - height / 2f - transform.padY) / transform.scale).coerceIn(0f, transform.sourceHeight.toFloat()),
                right = ((cx + width / 2f - transform.padX) / transform.scale).coerceIn(0f, transform.sourceWidth.toFloat()),
                bottom = ((cy + height / 2f - transform.padY) / transform.scale).coerceIn(0f, transform.sourceHeight.toFloat()),
                confidence = confidence,
            )
            if (candidate.right <= candidate.left || candidate.bottom <= candidate.top) continue
            candidates += candidate
        }
        val best = candidates.sortedByDescending { it.confidence }.firstOrNull { candidate ->
            candidates.none { other -> other.confidence > candidate.confidence && iou(candidate, other) > NMS_IOU }
        }
        if (best != null) Log.d(TAG, "cat candidate=${candidates.size} best=$best (normalized YOLO output scaled to ${inputWidth}x$inputHeight)")
        return best
    }

    private fun value(values: FloatArray, box: Int, channel: Int): Float = values[channel * outputCount + box]

    private fun iou(a: Detection, b: Detection): Float {
        val intersectionWidth = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val intersectionHeight = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val union = a.width * a.height + b.width * b.height - intersectionWidth * intersectionHeight
        return if (union <= 0f) 0f else intersectionWidth * intersectionHeight / union
    }

    override fun close() {
        interpreter.close()
    }

    private data class LetterboxTransform(
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val sourceWidth: Int,
        val sourceHeight: Int,
    )

    companion object {
        private const val TAG = "CatStalkerVision"
        private const val MODEL_FILE = "yolo11n_float32.tflite"
        private const val CAT_CLASS = 15
        private const val CONFIDENCE = 0.28f
        private const val NMS_IOU = 0.45f

        private fun loadModel(context: Context): MappedByteBuffer =
            context.assets.openFd(MODEL_FILE).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                    channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
                }
            }
    }
}
