#ifndef RANDOMFOREST_HPP
#define RANDOMFOREST_HPP

#include <string>
#include <tuple>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <codecvt>  // 用于宽字符转换
#include <filesystem>  // 添加文件系统支持
#include <numeric>  // 用于 std::accumulate
#include <Eigen/Dense>
#include <queue>
#include <fstream>
#include <string>
#include <map>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace Eigen;
class RandomForest {
public:
    // 成员变量
    int s;
    int proportion;
    vector<int> ssims;
    int result;
    vector<std::vector<float>> df;
    int index ;
    //string input_name;
    bool save = true;
    bool concat = true;
    cv::Mat firstmask, mask_, avgmask;
    cv::Mat firstrim, firstrim_gray, rim_;
    std::vector<int> bright_base = { 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255 };
    std::vector<int> count_1, count_2, count_3;
    int flag = 0;
    // 私有成员变量
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "RandomForest" };
    Ort::Session session{ nullptr };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    // ... 其他私有成员
    float g1, s1, percent1, k1, ks1, g2, s2, percent2, k2, ks2, g3, s3, percent3, k3, ks3;
    int frame = 1;
    int kcount = 0;
    float before_k = 0;
    float k_ = 0;
    float  k = 0;
    float cur_k = 0;
    float before_gray = 0;
    float k_init = 1;
    float weight = 0.95;
    float before_ssim = 0, before_ks = 0, ks = 0, cur_ks = 0;
    float kcountv = 0, before_v = 0, before_kv = 0, kv = 0, cur_kv = 0, kv_init = 1;
    int ks_init = 1;
    int kcounts;
    int concatcount = 0;
    int objectflag = 0;
    string csv_path;
    // ... 其他成员变量
    vector<float> currentgray, ssimlist, var;
    queue<float> k_queue;
    queue<float> ks_queue;
    queue<float> gray_queue;
    queue<float> percent_queue;
    queue<float> ssim_queue;
    // 成员函数
    RandomForest(const wchar_t* model_path);
    void initialize_model(const std::wstring& model_path);
    void initialize(cv::Mat cropped);
    int detect(const cv::Mat& cropped);
    void save_to_csv(const std::vector<std::vector<float>>& data, const std::string& filename);
    void init(string csv_path);
    string format_with_significant_digits(float value, int digits);
private:

    float roundToThreeSigFigs(float value);
    // 私有成员函数
    float state_estimate_concat(std::queue<float> k_queue,
        std::queue<float> ks_queue,
        std::queue<float> gray_queue,
        std::queue<float> percent_queue,
        std::queue<float> ssim_queue);
    float bright_cal_first(const cv::Mat img);
    std::pair<float, float> bright_cal2(const cv::Mat& ori, const cv::Mat& image,
        const cv::Mat& ori_mask, const cv::Mat& mask);
    float calculate_variance(const cv::Mat& image);
    float cal_var(const cv::Mat& img1, const cv::Mat& img2);
    float cal_ks(const std::vector<float>& ssims_, float s_);
    float cal_k(const std::vector<float>& grays_, float current_);
    std::tuple<std::vector<float>, float, float, std::vector<float>, float,
        std::vector<float>, float> filter(std::vector<float>& currentgray_,
            const std::vector<float>& ssimlist_,
            const std::vector<float>& varlist_);
    float ssim_semi(const cv::Mat& img_init, const cv::Mat& mask_init,
        const cv::Mat& img, const cv::Mat& mask);
    float structural_similarity(const cv::Mat& im1, const cv::Mat& im2,
        int win_size = 7, bool gradient = false,
        float data_range = 255.0, bool gaussian_weights = false,
        bool full = false, float K1 = 0.01, float K2 = 0.03,
        float sigma = 1.5, bool use_sample_covariance = true);
};

#endif // RANDOMFOREST_HPP