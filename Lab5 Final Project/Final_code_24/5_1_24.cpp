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
// 用於保護共享資料的鎖
std::mutex result_mutex;
std::mutex frame_mutex;

// 共享資料
vector<Object> global_objects;     // 最新的偵測結果
Mat ai_buffer;                     // 共享緩衝區
bool new_frame_ready_flag = false; // 告訴 AI 有新圖了
std::atomic<bool> is_running(true);// 程式是否繼續執行

// ================= 設定區域 =================
// 模型路徑 (請確保檔案在板子上)
const string MODEL_PROTO = "./best.opt.tnnproto";
const string MODEL_BIN   = "./best.opt.tnnmodel";
// YOLOv8 訓練時設定的參數
const int INPUT_WIDTH = 224;
const int INPUT_HEIGHT = 224;
const int NUM_CLASSES = 8;      // 題目要求的8類
float CONF_THRESHOLD = 0.45;    // 信心度門檻
float NMS_THRESHOLD = 0.45;     // 重疊過濾門檻
// 顯示設定
const int DISP_WIDTH = 640;
const int DISP_HEIGHT = 480;
// 類別名稱
const vector<string> CLASS_NAMES = {
    "banana", "book", "bottle", "cell phone", 
    "cup", "keyboard", "scissors", "spoon", 
};

// 執行 NMS 去除重疊框
// ================= NEON 加速版 NMS =================
void nms(vector<Object>& objects, float threshold) {
    if (objects.empty()) return;

    // 先排序 (機率高 -> 低)
    sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    int n = objects.size();
    
    // 轉換資料結構 (AoS -> SoA) 以便 NEON 平行載入
    // 需要連續的記憶體來存放座標，並轉成 float
    vector<float> x1(n), y1(n), x2(n), y2(n), areas(n);

    for (int i = 0; i < n; ++i) {
        x1[i] = (float)objects[i].rect.x;
        y1[i] = (float)objects[i].rect.y;
        x2[i] = (float)(objects[i].rect.x + objects[i].rect.width);
        y2[i] = (float)(objects[i].rect.y + objects[i].rect.height);
        areas[i] = (float)(objects[i].rect.width * objects[i].rect.height);
    }

    // 補齊到 4 的倍數，避免 NEON 讀取越界 (Padding)
    int padded_n = (n + 3) & ~3; 
    x1.resize(padded_n, 0); y1.resize(padded_n, 0);
    x2.resize(padded_n, 0); y2.resize(padded_n, 0);
    areas.resize(padded_n, 0);

    // NEON 平行計算核心
    // 準備全 0 向量 (用於計算 max(0, w))
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    // 準備 Threshold 向量
    float32x4_t v_thresh = vdupq_n_f32(threshold);

    for (int i = 0; i < n; ++i) {
        if (objects[i].prob == 0) continue; // 已經被刪掉的跳過

        // 載入當前最高分框 (Box A) 的參數，複製到所有通道
        float32x4_t a_x1 = vdupq_n_f32(x1[i]);
        float32x4_t a_y1 = vdupq_n_f32(y1[i]);
        float32x4_t a_x2 = vdupq_n_f32(x2[i]);
        float32x4_t a_y2 = vdupq_n_f32(y2[i]);
        float32x4_t a_area = vdupq_n_f32(areas[i]);

        // 檢查剩餘的框 (每次處理 4 個)
        for (int j = i + 1; j < padded_n; j += 4) {
            // 如果這 4 個索引都超過原始 n，就不用算了 (因為是 padding 的)
            if (j >= n) break;

            // 載入 4 個候選框 (Box B)
            float32x4_t b_x1 = vld1q_f32(&x1[j]);
            float32x4_t b_y1 = vld1q_f32(&y1[j]);
            float32x4_t b_x2 = vld1q_f32(&x2[j]);
            float32x4_t b_y2 = vld1q_f32(&y2[j]);
            float32x4_t b_area = vld1q_f32(&areas[j]);

            // 計算交集座標 (Intersection)
            // inter_x1 = max(a_x1, b_x1)
            float32x4_t inter_x1 = vmaxq_f32(a_x1, b_x1);
            float32x4_t inter_y1 = vmaxq_f32(a_y1, b_y1);
            // inter_x2 = min(a_x2, b_x2)
            float32x4_t inter_x2 = vminq_f32(a_x2, b_x2);
            float32x4_t inter_y2 = vminq_f32(a_y2, b_y2);

            // 計算交集寬高 w = max(0, inter_x2 - inter_x1)
            float32x4_t w = vmaxq_f32(vsubq_f32(inter_x2, inter_x1), v_zero);
            float32x4_t h = vmaxq_f32(vsubq_f32(inter_y2, inter_y1), v_zero);

            // 計算交集面積
            float32x4_t inter_area = vmulq_f32(w, h);

            // 計算聯集面積 (Union = A + B - Inter)
            float32x4_t union_area = vsubq_f32(vaddq_f32(a_area, b_area), inter_area);

            // 判斷 IoU > Threshold
            // 避免除法，改成判斷 Inter > Threshold * Union
            float32x4_t limit = vmulq_f32(union_area, v_thresh);
            // mask 結果: 符合條件的 bit 會全為 1 (NaN/Inf)，否則為 0
            uint32x4_t mask = vcgtq_f32(inter_area, limit);

            // 將結果存回陣列檢查
            // 直接把 4 個結果讀出來處理
            uint32_t res[4];
            vst1q_u32(res, mask);

            for (int k = 0; k < 4; ++k) {
                int cur_idx = j + k;
                if (cur_idx < n && res[k] != 0) { // res[k] != 0 代表 IoU > Threshold
                    objects[cur_idx].prob = 0; // 標記刪除
                }
            }
        }
    }
    
    // 移除被標記的框
    objects.erase(remove_if(objects.begin(), objects.end(), 
        [](const Object& obj) { return obj.prob == 0; }), objects.end());
}

// 輔助函式：計算 NC4HW4 格式的記憶體偏移量
// block_size 在 ARM 上通常是 4
inline int get_nc4hw4_index(int anchor_idx, int channel_idx, int area) {
    int block = channel_idx / 4;
    int remain = channel_idx % 4;
    // 公式：Block偏移 + Anchor偏移 + Channel內偏移
    return block * (area * 4) + anchor_idx * 4 + remain;
}

// YOLOv8 後處理核心邏輯
// 解析 TNN 輸出 Blob -> 轉換為 Object List
void postProcess(std::shared_ptr<tnn::Instance> instance, 
                 vector<Object>& objects, 
                 int frame_w, int frame_h) {
    objects.clear();

    // 取得 Raw Output Blob
    tnn::BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);
    if (output_blobs.empty()) return;

    tnn::Blob* output_blob = output_blobs.begin()->second;
    float* data = (float*)output_blob->GetHandle().base;
    
    // 參數設定
    int num_anchors  = 1029; // 224x224 (7x7 + 14x14 + 28x28)
    int num_classes  = 8;
    
    // 縮放比例
    float scale_x = (float)frame_w / INPUT_WIDTH;
    float scale_y = (float)frame_h / INPUT_HEIGHT;

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = 0.0f;
        int max_label = -1;

        // 讀取類別分數 (從 Channel 4 開始)
        for (int c = 0; c < num_classes; ++c) {
            // 使用 NC4HW4 公式取得正確位置
            int raw_index = get_nc4hw4_index(i, 4 + c, num_anchors);
            float score = data[raw_index];
            
            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        // 信心度過濾
        if (max_score > CONF_THRESHOLD) {
            // 讀取座標 (Channel 0, 1, 2, 3)
            // 直接用 NC4HW4 公式抓值
            float cx = data[get_nc4hw4_index(i, 0, num_anchors)];
            float cy = data[get_nc4hw4_index(i, 1, num_anchors)];
            float w  = data[get_nc4hw4_index(i, 2, num_anchors)];
            float h  = data[get_nc4hw4_index(i, 3, num_anchors)];

            // 轉回原圖座標
            float x = (cx - w * 0.5f) * scale_x;
            float y = (cy - h * 0.5f) * scale_y;
            float width = w * scale_x;
            float height = h * scale_y;

            // 邊界檢查
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
    nms(objects, NMS_THRESHOLD);
}

// ================= AI 工作執行緒 =================
void ai_worker_thread() {
    // 在執行緒內部初始化 TNN (每個執行緒獨立的 Instance 比較安全)
    string proto_content, model_content;
    
    // 讀取 Proto (網路結構)
    {
        ifstream proto_file(MODEL_PROTO);
        if (!proto_file.is_open()) { cerr << "Proto error!" << endl; return; }
        proto_content = string((istreambuf_iterator<char>(proto_file)), istreambuf_iterator<char>());
    }

    // 讀取 Model (權重)
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

    // 建立推論實例
    tnn::NetworkConfig network_config;
    network_config.device_type = tnn::DEVICE_ARM; // E9V3 是 ARM 架構
    network_config.enable_tune_kernel = true;     // false = 加速啟用階段
    tnn::Status status;
    auto instance = tnn_net.CreateInst(network_config, status);

    if (status != tnn::TNN_OK) {
        cerr << "[AI] Init failed!" << endl;
        return;
    }
    
    // 優化：設定 TNN 使用 2 核心
    instance->SetCpuNumThreads(2);

    Mat local_input_mat; // AI 執行緒私有的 Mat
    vector<Object> local_objects;

    while (is_running) {
        // 檢查有沒有新圖片
        bool has_job = false;
        {
            lock_guard<mutex> lock(frame_mutex);
            if (new_frame_ready_flag) {
                // 在鎖內進行複製，防止 datarace
                ai_buffer.copyTo(local_input_mat);
                has_job = true;
                new_frame_ready_flag = false;
            }
        }

        if (!has_job) {
            // 如果沒工作，稍微休息一下避免佔用 CPU 資源
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 開始推論 (約 0.8s，但不會卡住主畫面)
        // 預處理：使用 OpenCV 直接轉成 NCHW Float32
        // 自動完成：Resize(224x224) + SwapRB(BGR->RGB) + Normalize(0~1) + Layout(HWC->NCHW)
        // 不依賴 TNN 的轉換
        Mat blob;
        cv::dnn::blobFromImage(local_input_mat, blob, 
                               1.0 / 255.0,             // Scale: 壓縮到 0~1
                               Size(INPUT_WIDTH, INPUT_HEIGHT), // Resize
                               Scalar(0, 0, 0),         // Mean: 0
                               true,                    // SwapRB: BGR -> RGB
                               false);                  // Crop

        // 建立 TNN Mat，直接指向 blob 的資料
        // blob 是連續的 float 記憶體，格式為 NCHW
        std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
            tnn::DEVICE_ARM,
            tnn::NCHW_FLOAT, // 明確指定這是 NCHW 的 Float
            std::vector<int>{1, 3, INPUT_HEIGHT, INPUT_WIDTH}, 
            blob.data // 直接拿 OpenCV 算好的指標
        );

        // 因為已經在 blobFromImage 做過 Normalize 了
        // 用預設值就好
        tnn::MatConvertParam input_cvt_param;
        instance->SetInputMat(tnn_mat, input_cvt_param);
        
        // 推論
        instance->Forward();
        
        // 傳入的尺寸是 local_input_mat 的尺寸 (640x480)
        postProcess(instance, local_objects, local_input_mat.cols, local_input_mat.rows);

        // 更新全域結果
        {
            lock_guard<mutex> lock(result_mutex);
            global_objects = local_objects;
        }
    }
}

// ================= 主程式 =================
int main(int argc, char** argv) {
    // 參數解析
    if (argc > 1) {
        try {
            CONF_THRESHOLD = std::stof(argv[1]);
        } catch (...) {
            cerr << "Warning: Invalid Conf Threshold, using default." << endl;
        }
    }
    if (argc > 2) {
        try {
            NMS_THRESHOLD = std::stof(argv[2]);
        } catch (...) {
            cerr << "Warning: Invalid NMS Threshold, using default." << endl;
        }
    }

    // 暴力定頻
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

    // Camera Init
    VideoCapture cap(2);
    if (!cap.isOpened()) {
        cerr << "Cam Error" << endl;
        return -1;
    }
    
    // 關鍵優化：降低來源解析度
    // 其實可以不用，夠快了
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 預先配置記憶體，避免 copyTo 時重新 malloc
    ai_buffer.create(480, 640, CV_8UC3);

    thread ai_thread(ai_worker_thread);

    Mat capture_frame, display_frame;
    vector<Object> current_objects_to_draw;

    while (true) {
        cap >> capture_frame;
        if (capture_frame.empty()) break;

        // 嘗試送圖給 AI
        {
            // 使用 try_lock 避免卡住畫面
            if (frame_mutex.try_lock()) {
                capture_frame.copyTo(ai_buffer); 
                new_frame_ready_flag = true;
                frame_mutex.unlock();
            }
        }

        // 取得結果
        {
            lock_guard<mutex> lock(result_mutex);
            current_objects_to_draw = global_objects;
        }

        // 畫圖 (在 640x480 上畫)
        for (const auto& obj : current_objects_to_draw) {
            rectangle(capture_frame, obj.rect, Scalar(0, 255, 0), 2);
            string label = CLASS_NAMES[obj.label] + " " + to_string((int)(obj.prob * 100)) + "%";
            
            // 簡單的防止文字出界處理
            int y = obj.rect.y < 20 ? obj.rect.y + 20 : obj.rect.y - 5;
            putText(capture_frame, label, Point(obj.rect.x, y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        // 顯示到 Framebuffer
        // 顯示放大 (640x480 -> 640x480)
        // resize(capture_frame, display_frame, Size(DISP_WIDTH, DISP_HEIGHT));
        display_frame = capture_frame;
        
        // 關鍵：轉成 BGR565 (嵌入式螢幕常用格式)
        cvtColor(display_frame, display_frame, COLOR_BGR2BGR565);
        
        // 複製到 Framebuffer (逐行複製)
        // 假設螢幕也是 640x480，如果螢幕更大，這裡只會畫左上角
        int copy_w = min(width, DISP_WIDTH);
        int copy_h = min(height, DISP_HEIGHT);
        
        for (int y = 0; y < copy_h; y++) {
            // 計算偏移量 (16bit color)
            long location = (y * width) * 2;
            memcpy(fbp + location, display_frame.ptr(y), copy_w * 2);
        }

        // 離開檢查
        // 會錯誤，先註解掉
        // int key = waitKey(1); 
        // if (key == 27) { is_running = false; break; }
    }

    // 應該執行不到這裡
    is_running = false;
    if(ai_thread.joinable()) ai_thread.join();
    
    munmap(fbp, screensize);
    close(fb);
    return 0;
}