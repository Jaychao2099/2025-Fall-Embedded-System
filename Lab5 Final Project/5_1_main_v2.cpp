#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread> // 新增: 執行緒
#include <mutex>  // 新增: 互斥鎖
#include <atomic> // 新增: 原子變數
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

#include <opencv2/opencv.hpp>
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/mat_utils.h"

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
Mat global_frame_for_ai;           // 要給 AI 算的圖片
bool new_frame_ready = false;      // 告訴 AI 有新圖了
std::atomic<bool> is_running(true);// 程式是否繼續執行

// ================= 設定區域 =================
// 模型路徑 (請確保檔案在板子上)
const string MODEL_PROTO = "./best.opt.tnnproto";
const string MODEL_BIN   = "./best.opt.tnnmodel";
// YOLOv8 訓練時設定的參數
const int INPUT_WIDTH = 224;
const int INPUT_HEIGHT = 224;
const int NUM_CLASSES = 8;         // 題目要求的8類
const float CONF_THRESHOLD = 0.45; // 信心度門檻
const float NMS_THRESHOLD = 0.45;  // 重疊過濾門檻
// 顯示設定
const int DISP_WIDTH = 640;
const int DISP_HEIGHT = 480;
// 類別名稱
const vector<string> CLASS_NAMES = {
    "spoon", "banana", "keyboard", "cell phone", 
    "book", "scissors", "bottle", "cup"
};

// ================= 輔助函式 (NMS 與 後處理) =================

// 計算 IoU (Intersection over Union)
float get_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    return intersection / (area1 + area2 - intersection);
}

// 執行 NMS 去除重疊框
void nms(vector<Object>& objects, float threshold) {
    sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    for (size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].prob == 0) continue;
        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (objects[j].prob == 0) continue;
            if (get_iou(objects[i].rect, objects[j].rect) > threshold) {
                objects[j].prob = 0; // 標記為刪除
            }
        }
    }
    
    // 移除被標記的框
    objects.erase(remove_if(objects.begin(), objects.end(), 
        [](const Object& obj) { return obj.prob == 0; }), objects.end());
}

// YOLOv8 後處理核心邏輯
// 解析 TNN 輸出 Blob -> 轉換為 Object List
void postProcess(std::shared_ptr<tnn::Instance> instance, 
                 vector<Object>& objects, 
                 int frame_w, int frame_h) {
    
    objects.clear();

    tnn::BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);
    tnn::Blob* output_blob = output_blobs.begin()->second;
    
    float* data = (float*)output_blob->GetHandle().base;
    std::vector<int> shape = output_blob->GetBlobDesc().dims;
    
    // Debug: 第一次執行時可以打開看 Shape
    // printf("Shape: %d, %d, %d\n", shape[0], shape[1], shape[2]);

    // 判斷維度排列：[1, Channels, Anchors] 還是 [1, Anchors, Channels]
    bool is_channel_first = (shape[1] == (4 + NUM_CLASSES)); 
    
    int num_anchors = is_channel_first ? shape[2] : shape[1];
    int num_channels = is_channel_first ? shape[1] : shape[2];

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = 0.0f;
        int max_label = -1;
        
        // 取得該 Anchor 的資料指標或偏移量
        // 如果是 [1, 12, 1029]: data[channel * 1029 + i]
        // 如果是 [1, 1029, 12]: data[i * 12 + channel]

        // 1. 先找最大的 Class Score
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = 0.0f;
            if (is_channel_first) {
                score = data[(4 + c) * num_anchors + i];
            } else {
                score = data[i * num_channels + (4 + c)];
            }

            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        // 2. 只有信心度夠高才去算座標 (優化效能)
        if (max_score > CONF_THRESHOLD) {
            float cx, cy, w, h;
            
            if (is_channel_first) {
                cx = data[0 * num_anchors + i];
                cy = data[1 * num_anchors + i];
                w  = data[2 * num_anchors + i];
                h  = data[3 * num_anchors + i];
            } else {
                cx = data[i * num_channels + 0];
                cy = data[i * num_channels + 1];
                w  = data[i * num_channels + 2];
                h  = data[i * num_channels + 3];
            }

            // --- 關鍵修正：自動判斷座標單位 ---
            // 如果座標很小 (例如 < 2.0)，代表它是歸一化 (0~1) 的數據
            // 我們需要把它乘回 224 (INPUT_WIDTH)
            if (w <= 2.0f && h <= 2.0f) {
                cx *= INPUT_WIDTH;
                cy *= INPUT_HEIGHT;
                w  *= INPUT_WIDTH;
                h  *= INPUT_HEIGHT;
            }

            // 轉成左上角座標
            float x = cx - w / 2.0f;
            float y = cy - h / 2.0f;

            // 映射到實際螢幕大小 (640x480)
            int final_x = (int)(x / INPUT_WIDTH * frame_w);
            int final_y = (int)(y / INPUT_HEIGHT * frame_h);
            int final_w = (int)(w / INPUT_WIDTH * frame_w);
            int final_h = (int)(h / INPUT_HEIGHT * frame_h);

            // --- 安全性檢查：防止框框畫出界 ---
            final_x = max(0, min(final_x, frame_w - 1));
            final_y = max(0, min(final_y, frame_h - 1));
            final_w = min(final_w, frame_w - final_x);
            final_h = min(final_h, frame_h - final_y);
            
            if (final_w > 0 && final_h > 0) {
                Object obj;
                obj.rect = cv::Rect(final_x, final_y, final_w, final_h);
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
    // 1. 在執行緒內部初始化 TNN (每個執行緒獨立的 Instance 比較安全)
    // cout << "[AI Thread] Init TNN..." << endl;
    
    // 讀取 Proto (網路結構)
    string proto_content;
    {
        ifstream proto_file(MODEL_PROTO);
        if (!proto_file.is_open()) { cerr << "Proto file not found!" << endl; return; }
        proto_content = string((istreambuf_iterator<char>(proto_file)), istreambuf_iterator<char>());
    }

    // 讀取 Model (權重)
    string model_content;
    {
        ifstream model_file(MODEL_BIN, ios::binary);
        if (!model_file.is_open()) { cerr << "Model file not found!" << endl; return; }
        model_content = string((istreambuf_iterator<char>(model_file)), istreambuf_iterator<char>());
    }

    tnn::ModelConfig model_config;
    model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {proto_content, model_content};

    tnn::TNN tnn_net;
    tnn_net.Init(model_config);

    // 建立推論實例 (Instance)
    tnn::NetworkConfig network_config;
    network_config.device_type = tnn::DEVICE_ARM; // E9V3 是 ARM 架構
    network_config.enable_tune_kernel = true;     // 關閉 = 加速用啟用
    tnn::Status status;
    auto instance = tnn_net.CreateInst(network_config, status);

    if (status != tnn::TNN_OK) {
        cerr << "[AI Thread] Init failed!" << endl;
        return;
    }
    
    // cout << "[AI Thread] Ready loop." << endl;

    Mat input_mat;
    vector<Object> local_objects;

    while (is_running) {
        // 2. 檢查有沒有新圖片
        bool has_job = false;
        {
            lock_guard<mutex> lock(frame_mutex);
            if (new_frame_ready) {
                // 複製圖片過來，避免主執行緒修改
                input_mat = global_frame_for_ai.clone(); 
                new_frame_ready = false; // 標記已取走
                has_job = true;
            }
        }

        if (!has_job) {
            // 如果沒工作，稍微休息一下避免佔用 CPU 資源
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 3. 開始推論 (這裡會花 0.8s，但不會卡住主畫面)
        Mat input_blob_mat;
        resize(input_mat, input_blob_mat, Size(INPUT_WIDTH, INPUT_HEIGHT));
        cvtColor(input_blob_mat, input_blob_mat, COLOR_BGR2RGB);

        // 設定輸入
        // 使用 TNN 提供的工具將 OpenCV Mat 轉成 TNN Blob
        // 注意：這裡假設 input_blob_mat 是連續記憶體
        // 誠實告訴 TNN，傳進來的資料是 {1, 224, 224, 3} (Height, Width, Channel)
        // TNN 的 SetInputMat 會自動幫你把 HWC [224, 224, 3] 轉成模型需要的 NCHW [1, 3, 224, 224]
        std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
            tnn::DEVICE_ARM,
            tnn::N8UC3, 
            std::vector<int>{1, INPUT_HEIGHT, INPUT_WIDTH, 3}, // <--- 關鍵修改
            input_blob_mat.data
        );

        // instance->SetInputMat(tnn_mat, tnn::MatConvertParam());
        // 設定歸一化參數：(x - mean) * scale
        // YOLO 通常不需要 mean，但需要 scale 壓縮到 0~1
        tnn::MatConvertParam input_cvt_param;
        input_cvt_param.scale = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
        input_cvt_param.bias = {0.0, 0.0, 0.0};

        instance->SetInputMat(tnn_mat, input_cvt_param);
        
        // 推論
        instance->Forward();

        // 後處理
        // 注意：這裡傳入的是原始圖片的大小 (input_mat.cols)，不是 224
        postProcess(instance, local_objects, input_mat.cols, input_mat.rows);

        // 4. 更新全域結果
        {
            lock_guard<mutex> lock(result_mutex);
            global_objects = local_objects; // 把算好的框框丟出去
        }
        // cout << "[AI Thread] Updated objects: " << local_objects.size() << endl;
    }
}

// ================= 主程式 (顯示緒) =================
int main() {
    // Framebuffer Init
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) {
        cerr << "Error: Cannot open /dev/fb0" << endl;
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
        cerr << "Error: Cannot open camera 2" << endl;
        return -1;
    }

    // 啟動 AI 執行緒
    thread ai_thread(ai_worker_thread); // <--- 這裡啟動分身

    Mat frame, display_frame;
    vector<Object> current_objects_to_draw;

    // cout << "Main loop started..." << endl;

    while (true) {
        // 1. 讀取畫面 (快速)
        cap >> frame;
        if (frame.empty()) break;

        // 2. 傳送圖片給 AI (如果有空閒)
        {
            // 使用 try_lock 或者直接 lock，因為這裡很快
            lock_guard<mutex> lock(frame_mutex);
            if (!new_frame_ready) { // 如果 AI 還沒拿走上一張，我們就不要覆蓋 (或者你想覆蓋也可以)
                // 這裡選擇：如果 AI 正在忙，我們就更新成最新的圖，讓他一忙完就拿到最新的
                global_frame_for_ai = frame.clone(); // 必須 clone
                new_frame_ready = true;
            }
        }

        // 3. 取得目前最新的偵測結果
        {
            lock_guard<mutex> lock(result_mutex);
            current_objects_to_draw = global_objects; // 複製一份拿來畫
        }

        // 4. 畫圖 (畫在目前的 frame 上)
        // 注意：因為 frame 是新的，但 box 可能是 0.8 秒前的，
        // 所以物體移動太快時，框框會有點跟不上 (Lag)，這是正常的物理現象
        for (const auto& obj : current_objects_to_draw) {
            rectangle(frame, obj.rect, Scalar(0, 255, 0), 2);
            
            // 準備文字
            string label = CLASS_NAMES[obj.label] + " " + to_string((int)(obj.prob * 100)) + "%";
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            // 確保文字不會跑出上邊界 (如果 y 太小，就畫在框框裡面)
            int text_y = obj.rect.y - 5;
            if (text_y < labelSize.height) {
                text_y = obj.rect.y + labelSize.height + 5;
            }

            rectangle(frame, Point(obj.rect.x, text_y - labelSize.height),
                      Point(obj.rect.x + labelSize.width, text_y + baseLine),
                      Scalar(0, 255, 0), FILLED);
            
            putText(frame, label, Point(obj.rect.x, text_y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        // 5. 顯示到 Framebuffer
        resize(frame, display_frame, Size(DISP_WIDTH, DISP_HEIGHT));
        // 關鍵：轉成 BGR565 (嵌入式螢幕常用格式)
        cvtColor(display_frame, display_frame, COLOR_BGR2BGR565);
        
        // 寫入 mmap 記憶體
        // 假設是全螢幕左上角顯示
        for (int y = 0; y < DISP_HEIGHT; y++) {
            memcpy(fbp + ((0 + y) * width + 0) * 2, display_frame.ptr(y), DISP_WIDTH * 2);
        }

        // 6. 離開檢查
        // int key = waitKey(1); 
        // if (key == 27) { is_running = false; break; }
    }

    is_running = false;
    ai_thread.join(); // 等待 AI 執行緒結束
    
    // --- 清理 ---
    munmap(fbp, screensize);
    close(fb);
    cout << "Program exited." << endl;
    return 0;
}