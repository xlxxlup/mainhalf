#include "RandomForest.h"

RandomForest::RandomForest(const wchar_t* model_path) {
    index++;
    initialize_model(model_path);
}
double float_factorial(int n) {
    if (n < 0) return 0;
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}
void RandomForest::initialize(cv::Mat cropped) {
        frame++;
        if (frame == 1) {
            firstrim = cropped.clone();  // 使用OpenCV的clone方法复制图像
            // 保存分割后的图片，这部分取决于你的具体需求
        }
        else {
            rim_ = cropped.clone();
            // 保存分割后的图片
        }

        if (frame == 1) {
            float firstrim_gray = bright_cal_first(firstrim);  // 这里你需要提供bright_cal_first的具体实现
            float initial = firstrim_gray;
            // 如果有更多逻辑判断，请在此添加
            int s = 1;  // 根据上下文猜测s应该是一个整数
            currentgray.push_back(initial);
            ssimlist.push_back(s);
            var.push_back(cal_var(firstrim, firstrim));  // 提供cal_var的具体实现
        }
        else {
            float current, s;
            tie(current, s) = bright_cal2(firstrim, rim_, firstmask, mask_);  // 需要实现bright_cal2
            currentgray.push_back(current);
            ssimlist.push_back(s);
            var.push_back(cal_var(firstrim, rim_));
        }


        
    }
int RandomForest::detect(const cv::Mat& cropped) {
        if (objectflag == 1) {
            currentgray.push_back(0);
            ssimlist.push_back(0);
            k_queue.push(0);
            ks_queue.push(0);
            gray_queue.push(0);
            percent_queue.push(0);
            ssim_queue.push(0);
            return 3;
        }
        frame++;
        cout << "-------------------------------" << endl;
        cout << "开始检测 ： " << frame << endl;

        cv::Mat crop_ = cropped.clone();
        cv::Mat mask_ = avgmask;
        cv::Mat rim_ = cropped.clone();
        pair<float, float>current_s;
        float current, s;
        current_s = bright_cal2(firstrim, rim_, firstmask, mask_);
        currentgray.push_back(current_s.first);
        ssimlist.push_back(current_s.second);
        var.push_back(cal_var(firstrim, rim_));

        // SSIM、亮度平滑 并计算特征
        vector<float> grays, ssims;
        float proportion, v;
        tie(grays, current, proportion, ssims, s, var, v) = filter(currentgray, ssimlist, var);
        float k = cal_k(grays, current);  // 返回当前帧的变化率（平滑后）
        float ks = cal_ks(ssims, s);      // 返回当前帧的变化率（平滑后）

        if (k_queue.size() == 3 && ks_queue.size() == 3) {
            k_queue.pop();
            ks_queue.pop();
            gray_queue.pop();
            percent_queue.pop();
            ssim_queue.pop();
        }
        k_queue.push(k);
        ks_queue.push(ks);
        gray_queue.push(current);
        percent_queue.push(proportion);
        ssim_queue.push(s);

        int result;
        if (concat) {
            if (concatcount <= 1) {
                result = 0;
            }
            else {
                result = state_estimate_concat(k_queue, ks_queue, gray_queue, percent_queue, ssim_queue);
                cout << result << endl;
            }
        }

        if (save) {
            df.push_back({ current, proportion, s, k, ks, currentgray.back(), ssimlist.back() });
            save_to_csv(df, RandomForest::csv_path+".csv");
        }

        ++concatcount;

        return result;
    }
// 计算Savitzky-Golay系数
vector<double> savgol_coeffs(int window_length, int polyorder, int deriv = 0, double delta = 1.0, int pos = -1, string use = "conv") {
    if (polyorder >= window_length) {
        throw invalid_argument("polyorder must be less than window_length");
    }

    // 确定pos的默认值
    if (pos == -1) {
        if (window_length % 2 == 0) {
            pos = window_length / 2 - 0.5;
        }
        else {
            pos = window_length / 2;
        }
    }

    if (pos < 0 || pos >= window_length) {
        throw invalid_argument("pos must be within the window");
    }

    vector<double> x(window_length);
    for (int i = 0; i < window_length; ++i) {
        x[i] = i - pos;
    }

    if (use == "conv") {
        reverse(x.begin(), x.end());
    }

    // 构造矩阵A
    MatrixXd A(polyorder + 1, window_length);
    for (int k = 0; k <= polyorder; ++k) {
        for (int j = 0; j < window_length; ++j) {
            A(k, j) = pow(x[j], k);
        }
    }

    // 构造y向量
    VectorXd y = VectorXd::Zero(polyorder + 1);
    if (deriv <= polyorder) {
        y[deriv] = float_factorial(deriv) / pow(delta, deriv);
    }
    else {
        return vector<double>(window_length, 0.0);
    }

    // 求解最小二乘问题
    VectorXd coeffs = A.bdcSvd(ComputeThinU | ComputeThinV).solve(y);

    return vector<double>(coeffs.data(), coeffs.data() + coeffs.size());
}

// 多项式拟合
vector<double> polyfit(const vector<double>& t, const vector<double>& y, int order) {
    int rows = t.size();
    int cols = order + 1;
    MatrixXd A(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A(i, j) = pow(t[i], order - j);
        }
    }
    VectorXd b = Map<const VectorXd>(y.data(), rows);
    VectorXd coeffs = A.householderQr().solve(b);
    return vector<double>(coeffs.data(), coeffs.data() + cols);
}

// 多项式求导
vector<double> polyder(const vector<double>& p, int m) {
    if (m <= 0) return p;
    int current_degree = p.size() - 1;
    vector<double> dp = p;
    for (int k = 0; k < m; ++k) {
        if (current_degree < 0) break;
        for (int i = 0; i <= current_degree; ++i) {
            dp[i] *= (current_degree - i - k);
        }
        current_degree--;
        dp.resize(current_degree + 1);
    }
    return dp;
}

// 多项式求值
double polyval(const vector<double>& coeffs, double x) {
    if (coeffs.empty()) return 0.0;
    double result = 0.0;
    int degree = coeffs.size() - 1;
    for (size_t i = 0; i < coeffs.size(); ++i) {
        result += coeffs[i] * pow(x, degree - i);
    }
    return result;
}

// 处理边缘拟合
void fit_edge(const vector<double>& x, int window_start, int window_stop, int interp_start, int interp_stop, int polyorder, int deriv, double delta, vector<double>& y) {
    int window_size = window_stop - window_start;
    vector<double> x_window(window_size);
    for (int i = 0; i < window_size; ++i) {
        x_window[i] = x[window_start + i];
    }

    vector<double> t(window_size);
    for (int i = 0; i < window_size; ++i) {
        t[i] = i;
    }

    vector<double> coeffs = polyfit(t, x_window, polyorder);
    vector<double> coeffs_deriv = polyder(coeffs, deriv);

    for (int i = interp_start; i < interp_stop; ++i) {
        double ti = i - window_start;
        double val = polyval(coeffs_deriv, ti) / pow(delta, deriv);
        y[i] = val;
    }
}

// 卷积处理（简单实现，仅处理constant模式）
vector<double> convolve(const vector<double>& x, const vector<double>& coeffs, string mode, double cval) {
    int n = x.size();
    int window_length = coeffs.size();
    int halflen = window_length / 2;
    vector<double> y(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < window_length; ++j) {
            int pos = i + j - halflen;
            double val;
            if (pos < 0 || pos >= n) {
                val = cval;
            }
            else {
                val = x[pos];
            }
            sum += val * coeffs[j];
        }
        y[i] = sum;
    }
    return y;
}

// 主滤波函数
vector<double> savgol_filter(const vector<double>& x, int window_length, int polyorder, int deriv = 0, double delta = 1.0, string mode = "interp", double cval = 0.0) {
    if (polyorder >= window_length) {
        throw invalid_argument("polyorder must be less than window_length");
    }
    if (mode != "interp") {
        throw invalid_argument("Only 'interp' mode is currently supported");
    }

    vector<double> coeffs = savgol_coeffs(window_length, polyorder, deriv, delta);
    vector<double> y = convolve(x, coeffs, "constant", 0.0);

    int n = x.size();
    int halflen = window_length / 2;

    // 处理前边缘
    if (window_length <= n) {
        fit_edge(x, 0, window_length, 0, halflen, polyorder, deriv, delta, y);
        // 处理后边缘
        fit_edge(x, n - window_length, n, n - halflen, n, polyorder, deriv, delta, y);
    }

    return y;
}
void RandomForest::initialize_model(const wstring& model_path) {
    Ort::SessionOptions session_options;
    session = Ort::Session(env, model_path.c_str(), session_options);

    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    input_names.clear();
    output_names.clear();

    // 输入节点
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
    }

    // 输出节点
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    }
}

// 实现其他成员函数...

float RandomForest::bright_cal_first(const cv::Mat img) {
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    // 创建一个与灰度图像大小相同的掩码矩阵
    cv::Mat mask = grayImage != 0;

    // 计算非零像素的数量
    int n = cv::countNonZero(mask);

    if (n == 0) {
        cerr << "No non-zero pixels found in the image." << endl;
        return 0.0;
    }

    // 去掉黑色像素点后的数组
    cv::Mat totalPixelValue;
    grayImage.copyTo(totalPixelValue, mask);

    // 计算总像素值
    float sum = cv::sum(totalPixelValue)[0];

    // 计算平均像素值
    float averagePixelValue = sum / n;

    return averagePixelValue;
    // 实现计算亮度的逻辑
    // 返回平均灰度值

}

pair<float, float> RandomForest::bright_cal2(const cv::Mat& ori, const cv::Mat& image, const cv::Mat& ori_mask, const cv::Mat& mask) {
    // 创建一个临时图像
    cv::Mat temp = image.clone();
    cv::Mat temp1;
    // 将图像从BGR颜色空间转换为灰度颜色空间
    cv::cvtColor(image, temp1, cv::COLOR_BGR2GRAY);

    // 创建一个与灰度图像大小相同的掩码矩阵
    cv::Mat image1;
    cv::compare(temp1, 0, image1, cv::CMP_GT);  // image1 = image != 0

    // 计算非零像素的数量
    int n = cv::countNonZero(temp1);

    if (n == 0) {
        cerr << "No non-zero pixels found in the image." << endl;
        return { 0.0, 0.0 };
    }

    // 去掉黑色像素点后的数组
    cv::Mat total_pixel_value;
    temp1.copyTo(total_pixel_value, image1);

    // 计算总像素值
    float sum = cv::sum(total_pixel_value)[0];

    // 计算平均像素值
    float average_pixel_value = sum / n;

    // 计算SSIM
    float s = ssim_semi(ori, ori_mask, temp, mask);

    // 更新帧数


    return { average_pixel_value, s };
    // 实现计算两图之间的亮度及相似度等

}

// 实现剩余成员函数...

#include <iostream>
#include <fstream>
#include <iomanip>
#include <locale>
#include <vector>
#include <string>

using namespace std;

void RandomForest::save_to_csv(const vector<vector<float>>& data, const string& filename) {
    // 设置UTF-8编码
    locale utf8_locale(locale(), new codecvt_utf8<wchar_t>);
    ofstream file(filename);
    file.imbue(utf8_locale);

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    // 设置有效数字的位数（去掉 std::fixed）
    file << setprecision(3);

    // 写入表头
    file << " current, proportion, s, k, ks,currentgray,ssimlist\n";

    // 写入数据
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}
// 将浮点数四舍五入到三位有效数字
float RandomForest::roundToThreeSigFigs(float value) {
    // 四舍五入到小数点后6位
    float factor = 1e6f; // 10^6
    return round(value * factor) / factor;
}
float RandomForest::state_estimate_concat(queue<float> k_queue,
    queue<float> ks_queue,
    queue<float> gray_queue,
    queue<float> percent_queue,
    queue<float> ssim_queue) {
    // 从每个队列中取出头元素  
    g1 = gray_queue.front(); // 获取头元素但不移除  
    s1 = ssim_queue.front();
    percent1 = percent_queue.front();
    k1 = k_queue.front();
    ks1 = ks_queue.front();

    // 从队列中移除头元素  
    gray_queue.pop();
    ssim_queue.pop();
    percent_queue.pop();
    k_queue.pop();
    ks_queue.pop();

    g2 = gray_queue.front(); // 获取头元素但不移除  
    s2 = ssim_queue.front();
    percent2 = percent_queue.front();
    k2 = k_queue.front();
    ks2 = ks_queue.front();

    // 从队列中移除头元素  
    gray_queue.pop();
    ssim_queue.pop();
    percent_queue.pop();
    k_queue.pop();
    ks_queue.pop();

    g3 = gray_queue.front(); // 获取头元素但不移除  
    s3 = ssim_queue.front();
    percent3 = percent_queue.front();
    k3 = k_queue.front();
    ks3 = ks_queue.front();

    // 从队列中移除头元素  
    gray_queue.pop();
    ssim_queue.pop();
    percent_queue.pop();
    k_queue.pop();
    ks_queue.pop();



    // 准备输入数据
    std::vector<float> input_data = {
        roundToThreeSigFigs(g1), roundToThreeSigFigs(percent1), roundToThreeSigFigs(s1), roundToThreeSigFigs(k1), roundToThreeSigFigs(ks1),
        roundToThreeSigFigs(g2), roundToThreeSigFigs(percent2), roundToThreeSigFigs(s2), roundToThreeSigFigs(k2), roundToThreeSigFigs(ks2),
        roundToThreeSigFigs(g3), roundToThreeSigFigs(percent3), roundToThreeSigFigs(s3), roundToThreeSigFigs(k3), roundToThreeSigFigs(ks3)
    };
    
    vector<int64_t> input_dims = { 1, 15 };
    // 5. 创建输入Tensor

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_dims.data(),
        input_dims.size()
    );

    // 6. 获取输入名称（使用新API）
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_allocated = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_allocated.get();

    // 7. 获取输出名称（使用新API）
    Ort::AllocatedStringPtr output_name_allocated = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_allocated.get();

    // 8. 执行推理
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr },
        &input_name,    // 输入名称数组
        &input_tensor,  // 输入Tensor数组
        1,              // 输入数量
        &output_name,   // 输出名称数组
        1               // 输出数量
    );

    float* output_data;
    // 9. 解析输出结果
    if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
        output_data = output_tensors[0].GetTensorMutableData<float>();
        // 将浮点数转换为字符串
        std::ostringstream oss;
        oss << output_data[0];
        std::string value_str = oss.str();

        // 提取第一个数字字符
        char first_digit_char = value_str[0];
        int first_digit = first_digit_char - '0'; // 将字符转换为整数

        std::cout << "输出结果: " << first_digit << std::endl;
        return first_digit;
    }
    else {
        cerr << "错误：无法获取输出结果" << endl;
    }
    return 999;
}



float RandomForest::calculate_variance(const cv::Mat& image) {
    // 计算图像的均值
    cv::Scalar mean1, stddev1;

    cv::meanStdDev(image, mean1, stddev1);



    // 方差是标准差的平方  
    float variance1 = stddev1[0] * stddev1[0];


    return variance1;
}
float RandomForest::cal_var(const cv::Mat& img1, const cv::Mat& img2) {
    // 计算两个图像间的方差
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // 计算灰度图像的方差
    float var1 = calculate_variance(gray1);
    float var2 = calculate_variance(gray2);

    // 返回方差的绝对差值
    return abs(var1 - var2);
}

float RandomForest::cal_ks(const vector<float>& ssims_, float s_) {
    if (ks_init == 1) {
        ks_init += 1;
        // 对ssim34帧数组计算变化率
        for (float ssim_ : ssims_) {
            if (kcounts <= 1) {
                if (kcounts % 2 == 0) {
                    before_ssim = ssim_;
                }
                kcounts += 1;
            }
            else {
                if (kcounts % 2 == 0) {  // 间隔一帧取一点
                    cur_ks = ssim_ - before_ssim;  // 计算出当前未平滑斜率，并存储
                    ks = before_ks * weight + cur_ks * (1 - weight);  // EMA
                    // 更新属性
                    before_ssim = ssim_;
                    before_ks = ks;
                    kcounts += 1;
                }
                else {
                    ks = before_ks * weight + cur_ks * (1 - weight);
                    kcounts += 1;
                }
            }
        }
        return ks;
    }
    else {
        if (kcounts % 2 == 0) {  // 间隔一帧取一点
            cur_ks = s_ - before_ssim;  // 计算出当前未平滑斜率，并存储
            ks = before_ks * weight + cur_ks * (1 - weight);  // EMA
            // 更新属性
            before_ssim = s_;
            before_ks = ks;
            kcounts += 1;
            return ks;
        }
        else {
            ks = before_ks * weight + cur_ks * (1 - weight);
            kcounts += 1;
            return ks;
        }
    }
}

float RandomForest::cal_k(const vector<float>& grays_, float current_) {
    if (k_init == 1) {
        k_init += 1;
        // 对灰度34帧数组计算变化率
        for (float gray : grays_) {
            cout << "Processing gray: " << fixed << setprecision(16) << gray << "\n";
            if (kcount <= 1) {
                if (kcount % 2 == 0) {
                    before_gray = gray;
                    cout << "Setting before_gray to: " << fixed << setprecision(16) << before_gray << "\n";
                }
                kcount += 1;
            }
            else {
                if (kcount % 2 == 0) {  // 间隔一帧取一点
                    cur_k = gray - before_gray;  // 计算出当前未平滑斜率，并存储
                    k = before_k * weight + cur_k * (1 - weight);  // EMA
                    // 更新属性
                    before_gray = gray;
                    cout << "Updating before_gray to: " << fixed << setprecision(16) << before_gray << "\n";
                    before_k = k;
                    kcount += 1;
                }
                else {
                    k = before_k * weight + cur_k * (1 - weight);
                    kcount += 1;
                }
            }
        }
        return k;
    }
    else {
        if (kcount % 2 == 0) {  // 间隔一帧取一点
            cur_k = current_ - before_gray;  // 计算出当前未平滑斜率，并存储
            k = before_k * weight + cur_k * (1 - weight);  // EMA
            // 更新属性
            before_gray = current_;
            cout << "Updating before_gray to: " << fixed << setprecision(16) << before_gray << "\n";
            before_k = k;
            kcount += 1;
            return k;
        }
        else {
            k = before_k * weight + cur_k * (1 - weight);
            kcount += 1;
            return k;
        }
    }
}

tuple<vector<float>, float, float, vector<float>, float, vector<float>, float>
RandomForest::filter(vector<float>& currentgray_, const vector<float>& ssimlist_, const vector<float>& varlist_) {
    int window_length = 33;
    int order = 2;

    vector<double> graysoft, ssimsoft, varsoft;
    float percent;

    // Convert input vectors to double  
    vector<double> currentgray_double(currentgray_.size());
    transform(currentgray_.begin(), currentgray_.end(), currentgray_double.begin(),
        [](float val) { return static_cast<double>(val); });

    vector<double> ssimlist_double(ssimlist_.size());
    transform(ssimlist_.begin(), ssimlist_.end(), ssimlist_double.begin(),
        [](float val) { return static_cast<double>(val); });

    vector<double> varlist_double(varlist_.size());
    transform(varlist_.begin(), varlist_.end(), varlist_double.begin(),
        [](float val) { return static_cast<double>(val); });

    if (currentgray_.size() >= static_cast<size_t>(window_length) && ssimlist_.size() >= static_cast<size_t>(window_length)) {
        // Apply Savitzky-Golay filter  
        graysoft = savgol_filter(currentgray_double, window_length, order);
        ssimsoft = savgol_filter(ssimlist_double, window_length, order);
        varsoft = savgol_filter(varlist_double, window_length, order);
    }
    else {
        graysoft = currentgray_double;
        ssimsoft = ssimlist_double;
        varsoft = varlist_double;
    }

    if (!graysoft.empty()) {
        percent = (graysoft.back() - graysoft.front()) / graysoft.front();
    }
    else {
        throw runtime_error("空输入数据");
    }

    // Convert results back to float  
    vector<float> graysoft_float(graysoft.begin(), graysoft.end());
    vector<float> ssimsoft_float(ssimsoft.begin(), ssimsoft.end());
    vector<float> varsoft_float(varsoft.begin(), varsoft.end());

    return make_tuple(graysoft_float, static_cast<float>(graysoft.back()), percent,
        ssimsoft_float, static_cast<float>(ssimsoft.back()),
        varsoft_float, static_cast<float>(varsoft.back()));
}
float RandomForest::structural_similarity(
    const cv::Mat& im1,
    const cv::Mat& im2,
    int win_size, // 移除默认值
    bool gradient, // 移除默认值
    float data_range, // 移除默认值
    bool gaussian_weights, // 移除默认值
    bool full, // 移除默认值
    float K1, // 移除默认值
    float K2, // 移除默认值
    float sigma, // 移除默认值
    bool use_sample_covariance // 移除默认值
){
    // Check input images
    if (im1.empty() || im2.empty()) {
        cerr << "Input images are empty." << endl;
        return 0.0;
    }

    if (im1.size() != im2.size() || im1.type() != im2.type()) {
        cerr << "Input images must have the same size and type." << endl;
        return 0.0;
    }

    // Convert to doubleing point
    cv::Mat im1_double, im2_double;
    im1.convertTo(im1_double, CV_64F);
    im2.convertTo(im2_double, CV_64F);

    // Set window size
    if (win_size % 2 == 0) {
        cerr << "Window size must be odd." << endl;
        return 0.0;
    }

    // Compute means
    cv::Mat ux, uy;
    if (gaussian_weights) {
        cv::GaussianBlur(im1_double, ux, cv::Size(win_size, win_size), sigma);
        cv::GaussianBlur(im2_double, uy, cv::Size(win_size, win_size), sigma);
    }
    else {
        cv::boxFilter(im1_double, ux, -1, cv::Size(win_size, win_size), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
        cv::boxFilter(im2_double, uy, -1, cv::Size(win_size, win_size), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    }

    // Compute variances and covariances
    cv::Mat uxx, uyy, uxy;
    if (gaussian_weights) {
        cv::GaussianBlur(im1_double.mul(im1_double), uxx, cv::Size(win_size, win_size), sigma);
        cv::GaussianBlur(im2_double.mul(im2_double), uyy, cv::Size(win_size, win_size), sigma);
        cv::GaussianBlur(im1_double.mul(im2_double), uxy, cv::Size(win_size, win_size), sigma);
    }
    else {
        cv::boxFilter(im1_double.mul(im1_double), uxx, -1, cv::Size(win_size, win_size), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
        cv::boxFilter(im2_double.mul(im2_double), uyy, -1, cv::Size(win_size, win_size), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
        cv::boxFilter(im1_double.mul(im2_double), uxy, -1, cv::Size(win_size, win_size), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    }

    float NP = win_size * win_size;
    float cov_norm = use_sample_covariance ? NP / (NP - 1) : 1.0;

    cv::Mat vx = cov_norm * (uxx - ux.mul(ux));
    cv::Mat vy = cov_norm * (uyy - uy.mul(uy));
    cv::Mat vxy = cov_norm * (uxy - ux.mul(uy));

    // Constants
    float C1 = (K1 * data_range) * (K1 * data_range);
    float C2 = (K2 * data_range) * (K2 * data_range);

    // SSIM map
    cv::Mat A1, A2, B1, B2, S;
    A1 = 2 * ux.mul(uy) + C1;
    A2 = 2 * vxy + C2;
    B1 = ux.mul(ux) + uy.mul(uy) + C1;
    B2 = vx + vy + C2;
    S = (A1.mul(A2)) / (B1.mul(B2));

    // Crop the border to avoid edge effects
    int pad = (win_size - 1) / 2;
    cv::Rect roi(pad, pad, im1.cols - 2 * pad, im1.rows - 2 * pad);
    cv::Mat S_cropped = S(roi);

    // Compute mean SSIM
    float mssim = cv::mean(S_cropped)[0];

    if (gradient) {
        // Gradient computation (simplified for now)
        cv::Mat grad;
        if (full) {
            return mssim;  // Placeholder for full output
        }
        else {
            return mssim;  // Placeholder for gradient output
        }
    }
    else {
        if (full) {
            return mssim;  // Placeholder for full output
        }
        else {
            return mssim;
        }
    }

    return mssim;
}

float RandomForest::ssim_semi(const cv::Mat& img_init, const cv::Mat& mask_init, const cv::Mat& img, const cv::Mat& mask) {
    // 对图像进行缩放和平移操作，然后将结果转换为无符号8位整数类型
    cv::Mat img_init_converted, img_converted;
    img_init.convertTo(img_init_converted, CV_8U);
    img.convertTo(img_converted, CV_8U);

    // 获取图像的宽度和高度
    int width = img_init.cols;
    int height = img_init.rows;

    // 调整图像大小
    cv::resize(img_converted, img_converted, cv::Size(width, height));

    // 将图像从BGR颜色空间转换为灰度颜色空间
    cv::cvtColor(img_init_converted, img_init_converted, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_converted, img_converted, cv::COLOR_BGR2GRAY);


    // 计算SSIM

    float ssim_value = structural_similarity(img_init_converted, img_converted);

    return ssim_value;
}

void RandomForest::init(string csv_path) {
    s = 0;
    proportion = 0;
    result = 0;
    index = 0;
    firstmask = cv::Mat(); 
    mask_ =  cv::Mat(); 
    avgmask =  cv::Mat();;
    firstrim = cv::Mat();
    firstrim_gray = cv::Mat();
    rim_ = cv::Mat();
    // 清空向量和队列
    ssims.clear();
    df.clear();
    count_1.clear();
    count_2.clear();
    count_3.clear();
    currentgray.clear();
    ssimlist.clear();
    var.clear();
    RandomForest::csv_path = csv_path;
    while (!k_queue.empty()) k_queue.pop();
    while (!ks_queue.empty()) ks_queue.pop();
    while (!gray_queue.empty()) gray_queue.pop();
    while (!percent_queue.empty()) percent_queue.pop();
    while (!ssim_queue.empty()) ssim_queue.pop();
    g1 = 0;
    s1 = 0;
    percent1 = 0;
    k1 = 0;
    ks1 = 0;
    g2 = 0;
    s2 = 0;
    percent2 = 0;
    k2 = 0;
    ks2 = 0;
    g3 = 0;
    s3 = 0;
    percent3 = 0;
    k3 = 0;
    ks3 = 0;
    frame = 0;
    kcount = 0;
    kcounts = 0;
    concatcount = 0;
    objectflag = 0;
    before_k = 0;
    k_ = 0;
    k = 0;
    cur_k = 0;
    before_gray = 0;
    k_init = 1;
    weight = 0.95;
    before_ssim = 0;
    before_ks = 0;
    ks = 0;
    cur_ks = 0;
    kcountv = 0;
    before_v = 0;
    before_kv = 0;
    kv = 0;
    cur_kv = 0;
    kv_init = 1;
    ks_init = 1;
}

