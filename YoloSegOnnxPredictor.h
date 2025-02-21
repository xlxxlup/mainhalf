#ifndef YOLOSEG_ONNX_PREDICTOR_H
#define YOLOSEG_ONNX_PREDICTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
namespace fs = std::filesystem;  // 简化命名空间
class YoloSegOnnxPredictor {
public:
    // 构造函数
    YoloSegOnnxPredictor(const std::wstring& model_path, float conf_thres = 0.7, float iou_thres = 0.5, int num_masks = 32);

    // 对图像进行分割
    void segment_objects(const cv::Mat& image, std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids, std::vector<cv::Mat>& mask_maps);

private:
    // 初始化模型
    void initialize_model(const std::wstring& model_path);

    // 准备输入张量
    cv::Mat prepare_input(const cv::Mat& image);

    // 执行推理
    std::vector<Ort::Value> inference(cv::Mat& input_tensor);

    // 计算向量乘积
    size_t vectorProduct(const std::vector<int64_t> vector);

    // 预处理图像
    void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);

    // 调整图像大小
    void letterbox(const cv::Mat& image, cv::Mat& outImage, const cv::Size newShape = cv::Size(544, 544), const cv::Scalar color = cv::Scalar(114, 114, 114), bool auto_ = true, bool scaleUp = true, int stride = 32);

    // 处理输出
    void process_outputs(const std::vector<Ort::Value>& outputs,
        const cv::Size& original_size,
        std::vector<cv::Rect>& boxes,
        std::vector<float>& scores,
        std::vector<int>& class_ids,
        std::vector<cv::Mat>& mask_maps);

    // 环境和会话
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "YoloSegOnnxPredictor" };  // 使用宽字符
    Ort::Session session{ nullptr };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    int input_height;
    int input_width;
    float conf_threshold;
    float iou_threshold;
    int num_masks;
};

#endif // YOLOSEG_ONNX_PREDICTOR_H
