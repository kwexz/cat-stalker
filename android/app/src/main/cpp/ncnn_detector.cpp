#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <gpu.h>
#include <net.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace {
constexpr const char* kTag = "CatStalkerVision";
constexpr int kInputSize = 384;
constexpr int kCatClass = 0;
constexpr float kConfidence = 0.28f;
constexpr float kNmsIou = 0.45f;

struct Detection {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
};

struct NativeTiming {
    float preprocess_ms;
    float ncnn_ms;
    float postprocess_ms;
};

float iou(const Detection& a, const Detection& b);

float sample_plane(const unsigned char* plane, int row_stride, int pixel_stride, int width, int height, float x, float y) {
    x = std::clamp(x, 0.f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const float a = plane[y0 * row_stride + x0 * pixel_stride];
    const float b = plane[y0 * row_stride + x1 * pixel_stride];
    const float c = plane[y1 * row_stride + x0 * pixel_stride];
    const float d = plane[y1 * row_stride + x1 * pixel_stride];
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}

ncnn::Mat make_input_from_yuv420(const unsigned char* y, int y_row_stride, int y_pixel_stride, const unsigned char* u, int u_row_stride, int u_pixel_stride, const unsigned char* v, int v_row_stride, int v_pixel_stride, int width, int height) {
    const float scale = std::min(kInputSize / static_cast<float>(width), kInputSize / static_cast<float>(height));
    const int resized_width = static_cast<int>(width * scale);
    const int resized_height = static_cast<int>(height * scale);
    const int pad_left = (kInputSize - resized_width) / 2;
    const int pad_top = (kInputSize - resized_height) / 2;
    ncnn::Mat input(kInputSize, kInputSize, 3);
    input.fill(114.f);
    float* r = input.channel(0);
    float* g = input.channel(1);
    float* b = input.channel(2);
    for (int dy = 0; dy < resized_height; ++dy) {
        const float sy = (dy + .5f) / scale - .5f;
        for (int dx = 0; dx < resized_width; ++dx) {
            const float sx = (dx + .5f) / scale - .5f;
            const float luma = sample_plane(y, y_row_stride, y_pixel_stride, width, height, sx, sy);
            const float blue_chroma = sample_plane(u, u_row_stride, u_pixel_stride, width / 2, height / 2, sx / 2.f, sy / 2.f) - 128.f;
            const float red_chroma = sample_plane(v, v_row_stride, v_pixel_stride, width / 2, height / 2, sx / 2.f, sy / 2.f) - 128.f;
            const int pixel = (pad_top + dy) * kInputSize + pad_left + dx;
            r[pixel] = std::clamp(luma + 1.402f * red_chroma, 0.f, 255.f);
            g[pixel] = std::clamp(luma - .344136f * blue_chroma - .714136f * red_chroma, 0.f, 255.f);
            b[pixel] = std::clamp(luma + 1.772f * blue_chroma, 0.f, 255.f);
        }
    }
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(nullptr, norm);
    return input;
}

struct Detector {
    ncnn::Net net;
    bool gpu = false;

    explicit Detector(AAssetManager* assets) {
        ncnn::create_gpu_instance();
        gpu = ncnn::get_gpu_count() > 0;
        net.opt.num_threads = 4;
        net.opt.use_vulkan_compute = gpu;
        net.opt.use_fp16_packed = gpu;
        net.opt.use_fp16_storage = gpu;
        net.opt.use_fp16_arithmetic = gpu;
        if (net.load_param(assets, "p30_cat_v5_384_fp16_ncnn_model/model.ncnn.param") != 0 || net.load_model(assets, "p30_cat_v5_384_fp16_ncnn_model/model.ncnn.bin") != 0) {
            throw std::runtime_error("could not load NCNN assets");
        }
        __android_log_print(ANDROID_LOG_INFO, kTag, "NCNN initialized: Vulkan=%s", gpu ? "enabled" : "unavailable");
    }

    ~Detector() {
        net.clear();
        ncnn::destroy_gpu_instance();
    }
};

jfloatArray detect(Detector* detector, const ncnn::Mat& input, int width, int height, JNIEnv* env, NativeTiming* timing = nullptr) {
    const auto ncnn_started = std::chrono::steady_clock::now();
    const float scale = std::min(kInputSize / static_cast<float>(width), kInputSize / static_cast<float>(height));
    const int resized_width = static_cast<int>(width * scale);
    const int resized_height = static_cast<int>(height * scale);
    const int pad_left = (kInputSize - resized_width) / 2;
    const int pad_top = (kInputSize - resized_height) / 2;
    ncnn::Extractor extractor = detector->net.create_extractor();
    extractor.set_light_mode(true);
    if (extractor.input("in0", input) != 0) return nullptr;
    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0 || output.h < 5) return nullptr;
    const auto ncnn_finished = std::chrono::steady_clock::now();
    if (timing) timing->ncnn_ms = std::chrono::duration<float, std::milli>(ncnn_finished - ncnn_started).count();
    std::vector<Detection> candidates;
    const float* confidence = output.row(4 + kCatClass);
    for (int i = 0; i < output.w; ++i) {
        if (confidence[i] < kConfidence) continue;
        const float cx = output.row(0)[i];
        const float cy = output.row(1)[i];
        const float box_width = output.row(2)[i];
        const float box_height = output.row(3)[i];
        Detection candidate{
            std::clamp((cx - box_width * .5f - pad_left) / scale, 0.f, static_cast<float>(width)),
            std::clamp((cy - box_height * .5f - pad_top) / scale, 0.f, static_cast<float>(height)),
            std::clamp((cx + box_width * .5f - pad_left) / scale, 0.f, static_cast<float>(width)),
            std::clamp((cy + box_height * .5f - pad_top) / scale, 0.f, static_cast<float>(height)),
            confidence[i],
        };
        if (candidate.right > candidate.left && candidate.bottom > candidate.top) candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& higher : candidates) {
            if (higher.confidence <= candidate.confidence) break;
            if (iou(candidate, higher) > kNmsIou) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        const jfloat values[] = {candidate.left, candidate.top, candidate.right, candidate.bottom, candidate.confidence};
        auto result = env->NewFloatArray(5);
        env->SetFloatArrayRegion(result, 0, 5, values);
        if (timing) timing->postprocess_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - ncnn_finished).count();
        return result;
    }
    if (timing) timing->postprocess_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - ncnn_finished).count();
    return nullptr;
}

float iou(const Detection& a, const Detection& b) {
    const float width = std::max(0.f, std::min(a.right, b.right) - std::max(a.left, b.left));
    const float height = std::max(0.f, std::min(a.bottom, b.bottom) - std::max(a.top, b.top));
    const float intersection = width * height;
    const float total = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection;
    return total > 0.f ? intersection / total : 0.f;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_catstalker_vision_vision_NcnnDetector_create(JNIEnv* env, jobject, jobject asset_manager) {
    try {
        return reinterpret_cast<jlong>(new Detector(AAssetManager_fromJava(env, asset_manager)));
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "NCNN creation failed: %s", error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_catstalker_vision_vision_NcnnDetector_detect(JNIEnv* env, jobject, jlong handle, jobject bitmap) {
    auto* detector = reinterpret_cast<Detector*>(handle);
    if (!detector) return nullptr;

    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return nullptr;

    const float scale = std::min(kInputSize / static_cast<float>(info.width), kInputSize / static_cast<float>(info.height));
    const int resized_width = static_cast<int>(info.width * scale);
    const int resized_height = static_cast<int>(info.height * scale);
    const int pad_left = (kInputSize - resized_width) / 2;
    const int pad_top = (kInputSize - resized_height) / 2;

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return nullptr;
    ncnn::Mat rgb_pixels = ncnn::Mat::from_pixels_resize(static_cast<const unsigned char*>(pixels), ncnn::Mat::PIXEL_RGBA2RGB, info.width, info.height, resized_width, resized_height);
    AndroidBitmap_unlockPixels(env, bitmap);
    ncnn::Mat input;
    ncnn::copy_make_border(rgb_pixels, input, pad_top, kInputSize - resized_height - pad_top, pad_left, kInputSize - resized_width - pad_left, ncnn::BORDER_CONSTANT, 114.f);
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(nullptr, norm);

    return detect(detector, input, info.width, info.height, env);
}

extern "C" JNIEXPORT void JNICALL
Java_com_catstalker_vision_vision_NcnnDetector_destroy(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Detector*>(handle);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_catstalker_vision_vision_NcnnDetector_detectYuv(JNIEnv* env, jobject, jlong handle,
                                                          jobject y, jint y_row_stride, jint y_pixel_stride,
                                                          jobject u, jint u_row_stride, jint u_pixel_stride,
                                                          jobject v, jint v_row_stride, jint v_pixel_stride,
                                                          jint width, jint height) {
    auto* detector = reinterpret_cast<Detector*>(handle);
    const auto* y_bytes = static_cast<const unsigned char*>(env->GetDirectBufferAddress(y));
    const auto* u_bytes = static_cast<const unsigned char*>(env->GetDirectBufferAddress(u));
    const auto* v_bytes = static_cast<const unsigned char*>(env->GetDirectBufferAddress(v));
    if (!detector || !y_bytes || !u_bytes || !v_bytes) return nullptr;
    const auto preprocess_started = std::chrono::steady_clock::now();
    ncnn::Mat input = make_input_from_yuv420(y_bytes, y_row_stride, y_pixel_stride, u_bytes, u_row_stride, u_pixel_stride, v_bytes, v_row_stride, v_pixel_stride, width, height);
    NativeTiming timing{};
    timing.preprocess_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - preprocess_started).count();
    const auto result = detect(detector, input, width, height, env, &timing);
    static auto last_timing_log = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last_timing_log >= std::chrono::seconds(1)) {
        last_timing_log = now;
        __android_log_print(ANDROID_LOG_INFO, kTag, "native timing preprocess=%.1fms ncnn=%.1fms postprocess=%.1fms", timing.preprocess_ms, timing.ncnn_ms, timing.postprocess_ms);
    }
    return result;
}
