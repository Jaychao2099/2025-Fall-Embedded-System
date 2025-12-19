#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <fstream>

#include <opencv2/opencv.hpp>
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/mat_utils.h"

#include <arm_neon.h>

using namespace std;
using namespace cv;

// ================= 結構定義 =================
struct Object {
    cv::Rect rect;
    int label;
    float prob;
};

// ================= 全域變數與鎖 =================
std::mutex result_mutex;
std::mutex frame_mutex;

vector<Object> global_objects;     
Mat ai_buffer;                     
bool new_frame_ready_flag = false; 
std::atomic<bool> is_running(true);

// ================= 設定區域 =================
const string MODEL_PROTO = "./best.opt.tnnproto";
const string MODEL_BIN   = "./best.opt.tnnmodel";

const int INPUT_WIDTH = 224;
const int INPUT_HEIGHT = 224;
const int NUM_CLASSES = 8;         

// [修改 1] 移除單一 CONF_THRESHOLD，改用 Vector
// float CONF_THRESHOLD = 0.45; // 舊的
vector<float> g_class_thresholds(NUM_CLASSES, 0.45f); // 新的：預設每個類別都是 0.45

float NMS_THRESHOLD = 0.45;  

const int DISP_WIDTH = 640;
const int DISP_HEIGHT = 480;

const vector<string> CLASS_NAMES = {
    "banana", "book", "bottle", "cell phone", 
    "cup", "keyboard", "scissors", "spoon", 
};

// ================= NEON 加速版 NMS (無變更) =================
void nms(vector<Object>& objects, float threshold) {
    if (objects.empty()) return;
    sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    int n = objects.size();
    vector<float> x1(n), y1(n), x2(n), y2(n), areas(n);

    for (int i = 0; i < n; ++i) {
        x1[i] = (float)objects[i].rect.x;
        y1[i] = (float)objects[i].rect.y;
        x2[i] = (float)(objects[i].rect.x + objects[i].rect.width);
        y2[i] = (float)(objects[i].rect.y + objects[i].rect.height);
        areas[i] = (float)(objects[i].rect.width * objects[i].rect.height);
    }

    int padded_n = (n + 3) & ~3; 
    x1.resize(padded_n, 0); y1.resize(padded_n, 0);
    x2.resize(padded_n, 0); y2.resize(padded_n, 0);
    areas.resize(padded_n, 0);

    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_thresh = vdupq_n_f32(threshold);

    for (int i = 0; i < n; ++i) {
        if (objects[i].prob == 0) continue; 

        float32x4_t a_x1 = vdupq_n_f32(x1[i]);
        float32x4_t a_y1 = vdupq_n_f32(y1[i]);
        float32x4_t a_x2 = vdupq_n_f32(x2[i]);
        float32x4_t a_y2 = vdupq_n_f32(y2[i]);
        float32x4_t a_area = vdupq_n_f32(areas[i]);

        for (int j = i + 1; j < padded_n; j += 4) {
            if (j >= n) break;

            float32x4_t b_x1 = vld1q_f32(&x1[j]);
            float32x4_t b_y1 = vld1q_f32(&y1[j]);
            float32x4_t b_x2 = vld1q_f32(&x2[j]);
            float32x4_t b_y2 = vld1q_f32(&y2[j]);
            float32x4_t b_area = vld1q_f32(&areas[j]);

            float32x4_t inter_x1 = vmaxq_f32(a_x1, b_x1);
            float32x4_t inter_y1 = vmaxq_f32(a_y1, b_y1);
            float32x4_t inter_x2 = vminq_f32(a_x2, b_x2);
            float32x4_t inter_y2 = vminq_f32(a_y2, b_y2);

            float32x4_t w = vmaxq_f32(vsubq_f32(inter_x2, inter_x1), v_zero);
            float32x4_t h = vmaxq_f32(vsubq_f32(inter_y2, inter_y1), v_zero);

            float32x4_t inter_area = vmulq_f32(w, h);
            float32x4_t union_area = vsubq_f32(vaddq_f32(a_area, b_area), inter_area);
            float32x4_t limit = vmulq_f32(union_area, v_thresh);
            uint32x4_t mask = vcgtq_f32(inter_area, limit);

            uint32_t res[4];
            vst1q_u32(res, mask);

            for (int k = 0; k < 4; ++k) {
                int cur_idx = j + k;
                if (cur_idx < n && res[k] != 0) { 
                    objects[cur_idx].prob = 0; 
                }
            }
        }
    }
    
    objects.erase(remove_if(objects.begin(), objects.end(), 
        [](const Object& obj) { return obj.prob == 0; }), objects.end());
}

inline int get_nc4hw4_index(int anchor_idx, int channel_idx, int area) {
    int block = channel_idx / 4;
    int remain = channel_idx % 4;
    return block * (area * 4) + anchor_idx * 4 + remain;
}

// ================= [修改 2] PostProcess 使用個別門檻 =================
void postProcess(std::shared_ptr<tnn::Instance> instance, 
                 vector<Object>& objects, 
                 int frame_w, int frame_h) {
    objects.clear();

    tnn::BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);
    if (output_blobs.empty()) return;

    tnn::Blob* output_blob = output_blobs.begin()->second;
    float* data = (float*)output_blob->GetHandle().base;
    
    int num_anchors  = 1029; 
    int num_classes  = NUM_CLASSES;
    
    float scale_x = (float)frame_w / INPUT_WIDTH;
    float scale_y = (float)frame_h / INPUT_HEIGHT;

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = 0.0f;
        int max_label = -1;

        for (int c = 0; c < num_classes; ++c) {
            int raw_index = get_nc4hw4_index(i, 4 + c, num_anchors);
            float score = data[raw_index];
            
            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        // [關鍵修改] 根據 max_label 讀取對應的門檻值
        // 先確保 label 合法，再比較分數
        if (max_label >= 0 && max_label < NUM_CLASSES) {
            float current_class_threshold = g_class_thresholds[max_label];

            if (max_score > current_class_threshold) {
                float cx = data[get_nc4hw4_index(i, 0, num_anchors)];
                float cy = data[get_nc4hw4_index(i, 1, num_anchors)];
                float w  = data[get_nc4hw4_index(i, 2, num_anchors)];
                float h  = data[get_nc4hw4_index(i, 3, num_anchors)];

                float x = (cx - w * 0.5f) * scale_x;
                float y = (cy - h * 0.5f) * scale_y;
                float width = w * scale_x;
                float height = h * scale_y;

                int x1 = max(0, min((int)x, frame_w - 1));
                int y1 = max(0, min((int)y, frame_h - 1));
                int w1 = min((int)width, frame_w - x1);
                int h1 = min((int)height, frame_h - y1);
                if (w1 > 0 && h1 > 0) {
                    Object obj;
                    obj.rect = cv::Rect(x1, y1, w1, h1);
                    obj.label = max_label;
                    obj.prob = max_score;
                    objects.push_back(obj);
                }
            }
        }
    }
    nms(objects, NMS_THRESHOLD);
}

// ================= AI 工作執行緒 (無變更) =================
void ai_worker_thread() {
    string proto_content, model_content;
    
    {
        ifstream proto_file(MODEL_PROTO);
        if (!proto_file.is_open()) { cerr << "Proto error!" << endl; return; }
        proto_content = string((istreambuf_iterator<char>(proto_file)), istreambuf_iterator<char>());
    }

    {
        ifstream model_file(MODEL_BIN, ios::binary);
        if (!model_file.is_open()) { cerr << "Model error!" << endl; return; }
        model_content = string((istreambuf_iterator<char>(model_file)), istreambuf_iterator<char>());
    }

    tnn::ModelConfig model_config;
    model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {proto_content, model_content};

    tnn::TNN tnn_net;
    tnn_net.Init(model_config);

    tnn::NetworkConfig network_config;
    network_config.device_type = tnn::DEVICE_ARM; 
    network_config.enable_tune_kernel = true;     
    tnn::Status status;
    auto instance = tnn_net.CreateInst(network_config, status);

    if (status != tnn::TNN_OK) {
        cerr << "[AI] Init failed!" << endl;
        return;
    }
    
    instance->SetCpuNumThreads(2);

    Mat local_input_mat; 
    vector<Object> local_objects;

    while (is_running) {
        bool has_job = false;
        {
            lock_guard<mutex> lock(frame_mutex);
            if (new_frame_ready_flag) {
                ai_buffer.copyTo(local_input_mat);
                has_job = true;
                new_frame_ready_flag = false;
            }
        }

        if (!has_job) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        Mat blob;
        cv::dnn::blobFromImage(local_input_mat, blob, 
                               1.0 / 255.0,             
                               Size(INPUT_WIDTH, INPUT_HEIGHT), 
                               Scalar(0, 0, 0),         
                               true,                    
                               false);                  

        std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
            tnn::DEVICE_ARM,
            tnn::NCHW_FLOAT, 
            std::vector<int>{1, 3, INPUT_HEIGHT, INPUT_WIDTH}, 
            blob.data 
        );

        tnn::MatConvertParam input_cvt_param;
        instance->SetInputMat(tnn_mat, input_cvt_param);
        
        instance->Forward();
        
        postProcess(instance, local_objects, local_input_mat.cols, local_input_mat.rows);

        {
            lock_guard<mutex> lock(result_mutex);
            global_objects = local_objects;
        }
    }
}

// ================= 主程式 =================
int main(int argc, char** argv) {
    // [修改 3] 初始化門檻值邏輯
    float default_conf = 0.45;

    // 如果使用者有輸入 argv[1]，把它當作「預設值」套用到所有類別
    if (argc > 1) {
        try {
            default_conf = std::stof(argv[1]);
        } catch (...) {
            cerr << "Warning: Invalid Conf Threshold, using default 0.45." << endl;
        }
    }
    
    // 初始化所有類別為預設值
    std::fill(g_class_thresholds.begin(), g_class_thresholds.end(), default_conf);

    if (argc > 2) {
        try {
            NMS_THRESHOLD = std::stof(argv[2]);
        } catch (...) {
            cerr << "Warning: Invalid NMS Threshold, using default." << endl;
        }
    }

    // ==========================================================
    // [自定義區] 在這裡設定每個類別的個別門檻
    // 類別順序參考:
    // 0:banana, 1:book, 2:bottle, 3:cell phone, 
    // 4:cup, 5:keyboard, 6:scissors, 7:spoon
    // ==========================================================
    
    // === 範例：解除註解即可針對特定物品微調 ===
    // "banana", "book", "bottle", "cell phone", 
    // "cup", "keyboard", "scissors", "spoon", 
    g_class_thresholds[0] = 1.0;
    // g_class_thresholds[1] = 0.1;
    // g_class_thresholds[2] = 0.1;
    g_class_thresholds[3] = 1.0;
    // g_class_thresholds[4] = 0.01;
    // g_class_thresholds[5] = 0.3;
    g_class_thresholds[6] = 1.0;
    // g_class_thresholds[7] = 0.1;
    
    // 印出目前設定供檢查
    // cout << "=== Current Class Thresholds ===" << endl;
    // for(size_t i=0; i<CLASS_NAMES.size(); ++i) {
    //     cout << "ID " << i << " (" << CLASS_NAMES[i] << "): " << g_class_thresholds[i] << endl;
    // }
    // cout << "================================" << endl;


    // ---------------- 以下為硬體初始化 (無變更) ----------------
    system("echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) {
        cerr << "FB Error" << endl;
        return -1;
    }

    struct fb_var_screeninfo vinfo;
    if (ioctl(fb, FBIOGET_VSCREENINFO, &vinfo)) {
        cerr << "Error: Cannot get screen info" << endl;
        close(fb);
        return -1;
    }
    
    int width = vinfo.xres;
    int height = vinfo.yres;
    int bpp = vinfo.bits_per_pixel;
    int screensize = width * height * bpp / 8;

    unsigned char* fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
    if ((intptr_t)fbp == -1) {
        cerr << "Error: mmap failed" << endl;
        close(fb);
        return -1;
    }

    VideoCapture cap(2);
    if (!cap.isOpened()) {
        cerr << "Cam Error" << endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    ai_buffer.create(480, 640, CV_8UC3);

    thread ai_thread(ai_worker_thread);

    Mat capture_frame, display_frame;
    vector<Object> current_objects_to_draw;

    while (true) {
        cap >> capture_frame;
        if (capture_frame.empty()) break;

        {
            if (frame_mutex.try_lock()) {
                capture_frame.copyTo(ai_buffer); 
                new_frame_ready_flag = true;
                frame_mutex.unlock();
            }
        }

        {
            lock_guard<mutex> lock(result_mutex);
            current_objects_to_draw = global_objects;
        }

        for (const auto& obj : current_objects_to_draw) {
            rectangle(capture_frame, obj.rect, Scalar(0, 255, 0), 2);
            string label = CLASS_NAMES[obj.label] + " " + to_string((int)(obj.prob * 100)) + "%";
            
            int y = obj.rect.y < 20 ? obj.rect.y + 20 : obj.rect.y - 5;
            putText(capture_frame, label, Point(obj.rect.x, y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        display_frame = capture_frame;
        cvtColor(display_frame, display_frame, COLOR_BGR2BGR565);
        
        int copy_w = min(width, DISP_WIDTH);
        int copy_h = min(height, DISP_HEIGHT);
        
        for (int y = 0; y < copy_h; y++) {
            long location = (y * width) * 2;
            memcpy(fbp + location, display_frame.ptr(y), copy_w * 2);
        }

        // int key = waitKey(1); 
        // if (key == 27) { is_running = false; break; }
    }

    is_running = false;
    if(ai_thread.joinable()) ai_thread.join();
    
    munmap(fbp, screensize);
    close(fb);
    return 0;
}