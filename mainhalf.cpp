//#include <iostream>
//#include "YoloSegOnnxPredictor.h"
//#include "RandomForest.h"
//
//int extract_number(const std::string& filename) {
//    size_t dot_pos = filename.find_last_of('.');
//    std::string name_part = filename.substr(0, dot_pos);
//    size_t underscore_pos = name_part.find_last_of('_');
//    if (underscore_pos == std::string::npos) {
//        return -1; // 或抛出异常
//    }
//    std::string num_str = name_part.substr(underscore_pos + 1);
//    return std::stoi(num_str);
//}
//cv::Mat maskEnhance(const cv::Mat& mask) {
//    // 将三通道图像转换为单通道灰度图像
//    cv::Mat grayMask;
//    if (mask.channels() == 3) {
//        cv::cvtColor(mask, grayMask, cv::COLOR_BGR2GRAY);
//    }
//    else {
//        grayMask = mask.clone();
//    }
//
//    // 查找轮廓
//    std::vector<std::vector<cv::Point>> contours;
//    std::vector<cv::Vec4i> hierarchy;
//    cv::findContours(grayMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    // 创建空白掩膜
//    cv::Mat enhancedMask = cv::Mat::zeros(grayMask.size(), CV_8UC1);
//
//    if (!contours.empty()) {
//        // 找到最大轮廓
//        int maxIdx = 0;
//        double maxArea = cv::contourArea(contours[0]);
//        for (size_t i = 1; i < contours.size(); ++i) {
//            double area = cv::contourArea(contours[i]);
//            if (area > maxArea) {
//                maxArea = area;
//                maxIdx = i;
//            }
//        }
//
//        // 填充最大轮廓
//        cv::drawContours(enhancedMask, contours, maxIdx, cv::Scalar(255), cv::FILLED);
//    }
//
//
//    cv::Mat outputMask;
//    // 双边滤波进行平滑处理
//    try {
//        cv::bilateralFilter(enhancedMask, outputMask, 9, 75, 75);
//    }
//    catch (const cv::Exception& e) {
//        std::cerr << "OpenCV exception in bilateralFilter: " << e.what() << std::endl;
//    }
//
//    // 形态学闭操作进一步优化
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//    cv::morphologyEx(outputMask, outputMask, cv::MORPH_CLOSE, kernel);
//
//    return outputMask;
//}
//void generate_mask_image(const cv::Mat& image,
//    const std::vector<cv::Rect>& boxes,
//    const std::vector<cv::Mat>& masks,
//    cv::Mat& mask_image)
//{
//    // 创建单通道合并掩码
//    cv::Mat combined_mask = cv::Mat::zeros(image.size(), CV_8UC1);
//
//    // 合并所有掩码
//    for (const auto& mask : masks) {
//        cv::bitwise_or(combined_mask, mask, combined_mask);
//    }
//
//    // 转换为与原图相同的通道数（可选步骤）
//    if (image.channels() == 3) {
//        cv::cvtColor(combined_mask, mask_image, cv::COLOR_GRAY2BGR);
//    }
//    else {
//        combined_mask.copyTo(mask_image);
//    }
//
//    // 可视化增强：将掩码区域显示为白色
//    mask_image.setTo(cv::Scalar(255, 255, 255), combined_mask);
//}
//int main() {
//    //随机森林模型路径
//    const wchar_t* model_path1 = L"C:\\Users\\13679\\OneDrive\\Desktop\\推理\\RF_opaque_half_0_0_7.onnx";
//    //分割模型路径
//    wstring model_path = L"C:\\Users\\13679\\OneDrive\\Desktop\\推理\\test.onnx";
//    //特征csv保存路径
//    string csv_path = "C:\\Users\\13679\\OneDrive\\Desktop\\砂锅feature1";
//    // 原始小图路径
//    std::string input_folder = "D:\\Gong\\不透明视频\\现场视频原图"; 
//    //分割图保存路径
//    std::string output_folder = "C:\\Users\\13679\\OneDrive\\Desktop\\砂锅分割图1";
//
//
//    RandomForest processor(model_path1);
//    YoloSegOnnxPredictor predictor(model_path, 0.25);
//    // 创建输出目录（如果不存在）
//    fs::create_directories(output_folder);
//    fs::create_directories(csv_path); 
//    // 支持的图片格式扩展名
//    std::set<std::string> valid_exts = { ".jpg", ".jpeg", ".png", ".bmp" };
//
//    //帧数
//    int frame = 1;
//
//    //前五帧掩码图
//    std::vector<cv::Mat> masks;
//
//    //分割结果图  上一张掩码 平均掩码
//    cv::Mat  pre_masks, mean_masks, segmented_image;
//
//    for (const auto& subdir_entry : fs::directory_iterator(input_folder)) {
//        if (!subdir_entry.is_directory()) {
//            continue; // 跳过非目录项
//        }
//
//        // 准备输入输出路径
//        fs::path input_subdir = subdir_entry.path();
//        fs::path output_subdir = output_folder / input_subdir.filename();
//
//        fs::path csv_name = csv_path / input_subdir.filename();
//        fs::create_directories(output_subdir); // 创建输出子目录
//
//
//        // 子文件夹内处理变量初始化
//        int frame = 1;
//        cv::Mat pre_masks, mean_masks;
//        processor.init(csv_name.string());
//
//        std::vector<std::pair<int, fs::path>> file_list;
//        for (const auto& entry : fs::directory_iterator(input_subdir)) {
//            if (!valid_exts.count(entry.path().extension().string())) continue;
//
//            std::string filename = entry.path().filename().string();
//            int frame_num = extract_number(filename);
//            if (frame_num == -1) {
//                std::cerr << "跳过无法解析的文件: " << entry.path() << std::endl;
//                continue;
//            }
//            file_list.emplace_back(frame_num, entry.path());
//        }
//
//        // 按数字排序
//        std::sort(file_list.begin(), file_list.end(),
//            [](const auto& a, const auto& b) { return a.first < b.first; });
//
//        // 处理排序后的文件
//        for (const auto& [num, path] : file_list) {
//            cv::Mat image = cv::imread(path.string());
//            if (image.empty()) {
//                std::cerr << "无法读取图片: " << std::endl;
//                continue;
//            }
//
//            // 图像处理逻辑
//            std::vector<cv::Rect> boxes;
//            std::vector<float> scores;
//            std::vector<int> class_ids;
//            std::vector<cv::Mat> masks;
//            cv::Mat mask_finall, segmented_image;
//
//            try {
//                // 分割处理
//                if (frame <= 5) {
//                    // 创建第一帧图像的副本
//                    Mat image_aug = image.clone();
//                    // 对比度增强因子和像素偏移量
//                    double alpha = 1.3;  // 对比度增强因子
//                    int beta = -50;      // 像素偏移量
//                    // 应用线性变换（调整对比度和亮度）
//                    image_aug.convertTo(image_aug, -1, alpha, beta); // 调整对比度和亮度
//
//                    predictor.segment_objects(image_aug, boxes, scores, class_ids, masks);
//                    generate_mask_image(image, boxes, masks, mask_finall);
//                    cv::Mat enhanced_mask = maskEnhance(mask_finall);
//
//                    if (frame != 1) {
//                        cv::bitwise_and(enhanced_mask, pre_masks, mean_masks);
//                        cv::bitwise_and(image, image, segmented_image, enhanced_mask);
//                    }
//                    else {
//                        cv::bitwise_and(image, image, segmented_image, enhanced_mask);
//                    }
//                    pre_masks = enhanced_mask;
//                }
//                else {
//                    cv::bitwise_and(image, image, segmented_image, mean_masks);
//                }
//
//                // 处理图像序列
//                if (frame <= 33) {
//                    processor.initialize(segmented_image);
//                    // 保存结果到子目录
//                    std::string output_path = (output_subdir / input_subdir.filename()).string() + "_" + to_string(frame) + "_segmented.png";
//                    cv::imwrite(output_path, segmented_image);
//                    std::cout << "已处理: " << output_path << std::endl;
//
//                }
//                else {
//                    processor.detect(segmented_image);
//                }
//                frame++;
//
//
//            }
//            catch (const std::exception& e) {
//                std::cerr << "处理失败: " << "，错误: " << e.what() << std::endl;
//            }
//        }
//    }
//
//    std::cout << "批量处理完成！结果保存在: " << output_folder << std::endl;
//    return 0;
//}