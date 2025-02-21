//#include <iostream>
//#include "YoloSegOnnxPredictor.h"
//#include "RandomForest.h"
//
//int extract_number(const std::string& filename) {
//    size_t dot_pos = filename.find_last_of('.');
//    std::string name_part = filename.substr(0, dot_pos);
//    size_t underscore_pos = name_part.find_last_of('_');
//    if (underscore_pos == std::string::npos) {
//        return -1; // ���׳��쳣
//    }
//    std::string num_str = name_part.substr(underscore_pos + 1);
//    return std::stoi(num_str);
//}
//cv::Mat maskEnhance(const cv::Mat& mask) {
//    // ����ͨ��ͼ��ת��Ϊ��ͨ���Ҷ�ͼ��
//    cv::Mat grayMask;
//    if (mask.channels() == 3) {
//        cv::cvtColor(mask, grayMask, cv::COLOR_BGR2GRAY);
//    }
//    else {
//        grayMask = mask.clone();
//    }
//
//    // ��������
//    std::vector<std::vector<cv::Point>> contours;
//    std::vector<cv::Vec4i> hierarchy;
//    cv::findContours(grayMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    // �����հ���Ĥ
//    cv::Mat enhancedMask = cv::Mat::zeros(grayMask.size(), CV_8UC1);
//
//    if (!contours.empty()) {
//        // �ҵ��������
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
//        // ����������
//        cv::drawContours(enhancedMask, contours, maxIdx, cv::Scalar(255), cv::FILLED);
//    }
//
//
//    cv::Mat outputMask;
//    // ˫���˲�����ƽ������
//    try {
//        cv::bilateralFilter(enhancedMask, outputMask, 9, 75, 75);
//    }
//    catch (const cv::Exception& e) {
//        std::cerr << "OpenCV exception in bilateralFilter: " << e.what() << std::endl;
//    }
//
//    // ��̬ѧ�ղ�����һ���Ż�
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
//    // ������ͨ���ϲ�����
//    cv::Mat combined_mask = cv::Mat::zeros(image.size(), CV_8UC1);
//
//    // �ϲ���������
//    for (const auto& mask : masks) {
//        cv::bitwise_or(combined_mask, mask, combined_mask);
//    }
//
//    // ת��Ϊ��ԭͼ��ͬ��ͨ��������ѡ���裩
//    if (image.channels() == 3) {
//        cv::cvtColor(combined_mask, mask_image, cv::COLOR_GRAY2BGR);
//    }
//    else {
//        combined_mask.copyTo(mask_image);
//    }
//
//    // ���ӻ���ǿ��������������ʾΪ��ɫ
//    mask_image.setTo(cv::Scalar(255, 255, 255), combined_mask);
//}
//int main() {
//    //���ɭ��ģ��·��
//    const wchar_t* model_path1 = L"C:\\Users\\13679\\OneDrive\\Desktop\\����\\RF_opaque_half_0_0_7.onnx";
//    //�ָ�ģ��·��
//    wstring model_path = L"C:\\Users\\13679\\OneDrive\\Desktop\\����\\test.onnx";
//    //����csv����·��
//    string csv_path = "C:\\Users\\13679\\OneDrive\\Desktop\\ɰ��feature1";
//    // ԭʼСͼ·��
//    std::string input_folder = "D:\\Gong\\��͸����Ƶ\\�ֳ���Ƶԭͼ"; 
//    //�ָ�ͼ����·��
//    std::string output_folder = "C:\\Users\\13679\\OneDrive\\Desktop\\ɰ���ָ�ͼ1";
//
//
//    RandomForest processor(model_path1);
//    YoloSegOnnxPredictor predictor(model_path, 0.25);
//    // �������Ŀ¼����������ڣ�
//    fs::create_directories(output_folder);
//    fs::create_directories(csv_path); 
//    // ֧�ֵ�ͼƬ��ʽ��չ��
//    std::set<std::string> valid_exts = { ".jpg", ".jpeg", ".png", ".bmp" };
//
//    //֡��
//    int frame = 1;
//
//    //ǰ��֡����ͼ
//    std::vector<cv::Mat> masks;
//
//    //�ָ���ͼ  ��һ������ ƽ������
//    cv::Mat  pre_masks, mean_masks, segmented_image;
//
//    for (const auto& subdir_entry : fs::directory_iterator(input_folder)) {
//        if (!subdir_entry.is_directory()) {
//            continue; // ������Ŀ¼��
//        }
//
//        // ׼���������·��
//        fs::path input_subdir = subdir_entry.path();
//        fs::path output_subdir = output_folder / input_subdir.filename();
//
//        fs::path csv_name = csv_path / input_subdir.filename();
//        fs::create_directories(output_subdir); // ���������Ŀ¼
//
//
//        // ���ļ����ڴ��������ʼ��
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
//                std::cerr << "�����޷��������ļ�: " << entry.path() << std::endl;
//                continue;
//            }
//            file_list.emplace_back(frame_num, entry.path());
//        }
//
//        // ����������
//        std::sort(file_list.begin(), file_list.end(),
//            [](const auto& a, const auto& b) { return a.first < b.first; });
//
//        // �����������ļ�
//        for (const auto& [num, path] : file_list) {
//            cv::Mat image = cv::imread(path.string());
//            if (image.empty()) {
//                std::cerr << "�޷���ȡͼƬ: " << std::endl;
//                continue;
//            }
//
//            // ͼ�����߼�
//            std::vector<cv::Rect> boxes;
//            std::vector<float> scores;
//            std::vector<int> class_ids;
//            std::vector<cv::Mat> masks;
//            cv::Mat mask_finall, segmented_image;
//
//            try {
//                // �ָ��
//                if (frame <= 5) {
//                    // ������һ֡ͼ��ĸ���
//                    Mat image_aug = image.clone();
//                    // �Աȶ���ǿ���Ӻ�����ƫ����
//                    double alpha = 1.3;  // �Աȶ���ǿ����
//                    int beta = -50;      // ����ƫ����
//                    // Ӧ�����Ա任�������ԱȶȺ����ȣ�
//                    image_aug.convertTo(image_aug, -1, alpha, beta); // �����ԱȶȺ�����
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
//                // ����ͼ������
//                if (frame <= 33) {
//                    processor.initialize(segmented_image);
//                    // ����������Ŀ¼
//                    std::string output_path = (output_subdir / input_subdir.filename()).string() + "_" + to_string(frame) + "_segmented.png";
//                    cv::imwrite(output_path, segmented_image);
//                    std::cout << "�Ѵ���: " << output_path << std::endl;
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
//                std::cerr << "����ʧ��: " << "������: " << e.what() << std::endl;
//            }
//        }
//    }
//
//    std::cout << "����������ɣ����������: " << output_folder << std::endl;
//    return 0;
//}