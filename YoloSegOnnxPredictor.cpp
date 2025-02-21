#include "YoloSegOnnxPredictor.h"

YoloSegOnnxPredictor::YoloSegOnnxPredictor(const std::wstring& model_path, float conf_thres, float iou_thres, int num_masks)
    : conf_threshold(conf_thres), iou_threshold(iou_thres), num_masks(num_masks) {
    initialize_model(model_path);  // 调用 initialize_model 来初始化 session
}

void YoloSegOnnxPredictor::segment_objects(const cv::Mat& image, std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids, std::vector<cv::Mat>& mask_maps) {
    cv::Mat input_tensor = prepare_input(image);
    std::vector<Ort::Value> outputs = inference(input_tensor);
    process_outputs(outputs, image.size(), boxes, scores, class_ids, mask_maps);
}

void YoloSegOnnxPredictor::initialize_model(const std::wstring& model_path) {
    Ort::SessionOptions session_options;
    session = Ort::Session(env, model_path.c_str(), session_options);  // 使用 env 和 model_path 初始化 session

    // 获取输入和输出节点名称
    Ort::AllocatorWithDefaultOptions allocator;
    input_names.clear();
    output_names.clear();

    // 获取输入节点名称
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
    }

    // 获取输出节点名称
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    }

    // 获取输入张量的形状
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_height = input_shape[2];
    input_width = input_shape[3];
}

cv::Mat YoloSegOnnxPredictor::prepare_input(const cv::Mat& image) {
    // 转换为 RGB 并归一化
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    cv::Mat resized_image;
    cv::resize(rgb_image, resized_image, cv::Size(input_width, input_height));
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // HWC 转 CHW
    cv::Mat channels[3];
    cv::split(resized_image, channels);
    cv::Mat input_tensor;
    cv::merge(channels, 3, input_tensor);

    return input_tensor;
}

std::vector<Ort::Value> YoloSegOnnxPredictor::inference(cv::Mat& input_tensor) {
    float* blob = nullptr;
    std::vector<int64_t> inputTensorShape{ 1,3,-1,-1 };
    preprocessing(input_tensor, blob, inputTensorShape);

    size_t inputTensorSize = vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> input_shape = { 1, 3, input_height, input_width };

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    inputNames.push_back("images");
    outputNames.push_back("output0");
    outputNames.push_back("output1");
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, input_shape.data(), input_shape.size()));

    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{ nullptr },
        inputNames.data(), inputTensors.data(), 1, outputNames.data(), 2);

    return outputTensors;
}

size_t YoloSegOnnxPredictor::vectorProduct(const std::vector<int64_t> vector) {
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector) {
        product *= element;
    }
    return product;
}

void YoloSegOnnxPredictor::preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape) {
    cv::Mat resizedImage, floatImage; // 尺寸调整，类型转换
    resizedImage = image.clone();
    bool  isDynamicInputShape = false;
    const cv::Size inputSize = cv::Size(544, 544);
    cv::Size2f inputImageShape = cv::Size2f(inputSize);
    letterbox(resizedImage, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    floatImage = resizedImage.clone();
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];

    cv::Size floatImageSize{ floatImage.cols,floatImage.rows };
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

void YoloSegOnnxPredictor::letterbox(const cv::Mat& image, cv::Mat& outImage, const cv::Size newShape, const cv::Scalar color, bool auto_, bool scaleUp, int stride) {
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    int newUnpad[2] = { (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };
    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void YoloSegOnnxPredictor::process_outputs(const std::vector<Ort::Value>& outputs,
    const cv::Size& original_size,
    std::vector<cv::Rect>& boxes,
    std::vector<float>& scores,
    std::vector<int>& class_ids,
    std::vector<cv::Mat>& mask_maps) {
    const Ort::Value& detections = outputs[0];
    const Ort::Value& proto = outputs[1];

    const float* detections_data = detections.GetTensorData<float>();
    const float* proto_data = proto.GetTensorData<float>();

    auto detections_shape = detections.GetTensorTypeAndShapeInfo().GetShape();
    auto proto_shape = proto.GetTensorTypeAndShapeInfo().GetShape();

    const int64_t num_rows = detections_shape[1];
    const int64_t num_columns = detections_shape[2];
    const int64_t proto_channels = proto_shape[1];
    const int64_t proto_height = proto_shape[2];
    const int64_t proto_width = proto_shape[3];

    std::vector<cv::Rect> raw_boxes;
    std::vector<float> raw_confidences;
    std::vector<int> raw_class_ids;
    std::vector<std::vector<float>> mask_coeffs_list;

    for (int64_t k = 0; k < 6069; ++k) {
        const float* detection = detections_data;

        const float x_center = detection[k];
        const float y_center = detection[k + 1 * num_columns + 0];
        const float width = detection[k + 2 * num_columns + 0];
        const float height = detection[k + 3 * num_columns + 0];
        const float obj_score = detection[k + 4 * num_columns + 0];

        const float cls_score = 1.0f;
        const float total_score = obj_score * cls_score;

        if (total_score < conf_threshold) continue;

        std::vector<float> mask_coeffs;
        for (int i = 0; i < 32; i++) {
            mask_coeffs.push_back(detection[k + (i + 5) * num_columns + 0]);
        }

        const float scale_x = original_size.width / static_cast<float>(input_width);
        const float scale_y = original_size.height / static_cast<float>(input_height);

        float x_center_r = x_center * scale_x;
        float y_center_r = y_center * scale_y;
        float width_r = width * scale_x;
        float height_r = height * scale_y;

        float x1 = (x_center_r - width_r / 2);
        float y1 = (y_center_r - height_r / 2);
        float x2 = (x_center_r + width_r / 2);
        float y2 = (y_center_r + height_r / 2);

        x1 = std::clamp(x1, 0.0f, static_cast<float>(original_size.width));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(original_size.height));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(original_size.width));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(original_size.height));

        cv::Rect raw_box(
            static_cast<float>(x1),
            static_cast<float>(y1),
            static_cast<float>(x2 - x1),
            static_cast<float>(y2 - y1)
        );
        raw_boxes.emplace_back(raw_box);

        /*raw_boxes.emplace_back(
            static_cast<int>(x1),
            static_cast<int>(y1),
            static_cast<int>(x2 - x1),
            static_cast<int>(y2 - y1)
        );*/
        boxes.push_back(raw_box);
        scores.push_back(total_score);
        class_ids.push_back(0);
        mask_coeffs_list.emplace_back(mask_coeffs);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    std::vector<cv::Rect> final_boxes;
    std::vector<float> final_scores;
    std::vector<int> final_class_ids;
    std::vector<cv::Mat> final_masks;

    cv::Mat proto_matrix(proto_channels, proto_height * proto_width, CV_32F, const_cast<float*>(proto_data));

    for (const int idx : indices) {
        final_boxes.push_back(boxes[idx]);
        final_scores.push_back(scores[idx]);
        final_class_ids.push_back(class_ids[idx]);

        const auto& coeffs = mask_coeffs_list[idx];
        cv::Mat coeff_mat(1, proto_channels, CV_32F, const_cast<float*>(coeffs.data()));
        cv::Mat mask_mat = coeff_mat * proto_matrix;

        mask_mat = mask_mat.reshape(1, proto_height);
        cv::exp(-mask_mat, mask_mat);
        mask_mat = 1.0 / (1.0 + mask_mat);
        float width = original_size.width;
        float height = original_size.height;
        const float scale_x2 = proto_width / width;
        const float scale_y2 = proto_height / height;
        const cv::Rect& rect1 = final_boxes.back();
        float x1_2 = rect1.x;
        float y1_2 = rect1.y;
        float x2_2 = rect1.width + x1_2;
        float y2_2 = rect1.height + y1_2;
        int x1_r = x1_2 * scale_x2;
        int y1_r = y1_2 * scale_y2;
        int x2_r = x2_2 * scale_x2;
        int y2_r = y2_2 * scale_y2;
        cv::Rect box(cv::Point(x1_r, y1_r), cv::Point(x2_r, y2_r));


        /*cv::Mat mask_upsampled;
        cv::resize(mask_mat, mask_upsampled, cv::Size(input_width, input_height));*/

        const cv::Rect& roi = box;
        cv::Mat mask_cropped = mask_mat(roi);

        cv::Mat mask_resized;
        cv::resize(mask_cropped, mask_resized, cv::Size(x2_2 - x1_2, y2_2 - y1_2), 0, 0, cv::INTER_CUBIC);

        cv::Mat binary_mask;
        cv::threshold(mask_resized, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        // 假设 mask_maps 是一个与原图大小相同的全黑图
        cv::Mat mask_maps1 = cv::Mat::zeros(height, width, CV_8UC1); // 单通道灰度图
        // 坐标
        // 将 crop_mask 赋值到 mask_maps 的指定区域
        cv::Mat roi2 = mask_maps1(cv::Range(y1_2, y2_2), cv::Range(x1_2, x2_2)); // 创建 ROI
        binary_mask.copyTo(roi2); // 将 crop_mask 复制到 ROI


        final_masks.push_back(mask_maps1);
    }

    boxes = std::move(final_boxes);
    scores = std::move(final_scores);
    class_ids = std::move(final_class_ids);
    mask_maps = std::move(final_masks);
}
