#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring> // for memcpy

// OpenCV Headers
#include <opencv2/opencv.hpp>

// TNN Headers (請確保你的 Makefile 有指到 include 路徑)
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/mat_utils.h"

using namespace std;
using namespace cv;

// ================= 設定區域 =================
// 模型路徑 (請確保檔案在板子上)
const string MODEL_PROTO = "./best.opt.tnnproto";
const string MODEL_BIN   = "./best.opt.tnnmodel";

// YOLOv8 訓練時設定的參數
const int INPUT_WIDTH = 224;
const int INPUT_HEIGHT = 224;
const int NUM_CLASSES = 8; // 題目要求的8類
const float CONF_THRESHOLD = 0.45; // 信心度門檻
const float NMS_THRESHOLD = 0.45;  // 重疊過濾門檻

// 顯示設定 (配合你的 Lab3 設定)
const int DISP_WIDTH = 640;
const int DISP_HEIGHT = 480;

// 類別名稱
const vector<string> CLASS_NAMES = {
    "spoon", "banana", "keyboard", "cell phone", 
    "book", "scissors", "bottle", "cup"
};

// ================= 結構定義 =================
struct Object {
    cv::Rect rect;
    int label;
    float prob;
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
    
    // 1. 取得輸出 Blob (名稱通常是 output0，如果不確定可用 GetAllOutputBlobs)
    tnn::BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);
    tnn::Blob* output_blob = output_blobs.begin()->second;

    // 2. 取得資料指針
    float* data = (float*)output_blob->GetHandle().base;
    tnn::BlobShape shape = output_blob->GetBlobDesc().dims;
    
    // YOLOv8 輸出形狀通常是 [Batch, 4+Classes, Anchors]
    // 對於 224x224, nc=8: [1, 12, 1029] (1029 = 224/8^2 + ...)
    int num_channels = shape[1]; // 12 (4個座標 + 8個類別)
    int num_anchors = shape[2];  // 1029

    objects.clear();

    // 3. 遍歷所有 Anchor
    // 注意：TNN 輸出的記憶體排列可能是連續的，YOLOv8 輸出通常需要轉置讀取
    // 這裡假設輸出格式為 [1, channels, anchors]，即 data[channel * num_anchors + anchor_idx]
    
    for (int i = 0; i < num_anchors; ++i) {
        // 找出該 Anchor 中分數最高的類別
        float max_score = 0.0f;
        int max_label = -1;

        // 類別分數從第 4 行開始 (0,1,2,3 是座標)
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        // 如果分數夠高，就解析座標
        if (max_score > CONF_THRESHOLD) {
            // YOLOv8 輸出是 cx, cy, w, h (相對於 224x224 的數值)
            float cx = data[0 * num_anchors + i];
            float cy = data[1 * num_anchors + i];
            float w  = data[2 * num_anchors + i];
            float h  = data[3 * num_anchors + i];

            // 轉換成左上角座標 (x, y)
            float x = cx - w / 2.0f;
            float y = cy - h / 2.0f;

            // 還原回原始攝影機畫面的比例
            // 因為模型輸入是 224x224，但畫面顯示可能是 640x480
            // 這裡我們先存正規化座標，畫圖時再乘回去比較準
            Object obj;
            obj.rect = cv::Rect(
                (int)(x / INPUT_WIDTH * frame_w),
                (int)(y / INPUT_HEIGHT * frame_h),
                (int)(w / INPUT_WIDTH * frame_w),
                (int)(h / INPUT_HEIGHT * frame_h)
            );
            obj.label = max_label;
            obj.prob = max_score;
            objects.push_back(obj);
        }
    }

    // 4. NMS 過濾
    nms(objects, NMS_THRESHOLD);
}

// ================= 主程式 =================
int main() {
    // ------------------- 1. Framebuffer 初始化 (同 Lab3) -------------------
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
    
    cout << "Framebuffer Init OK: " << width << "x" << height << " " << bpp << "bpp" << endl;

    // ------------------- 2. TNN 模型初始化 -------------------
    cout << "Loading TNN Model..." << endl;
    
    // 讀取 Proto (網路結構)
    string proto_content;
    {
        ifstream proto_file(MODEL_PROTO);
        if (!proto_file.is_open()) { cerr << "Proto file not found!" << endl; return -1; }
        proto_content = string((istreambuf_iterator<char>(proto_file)), istreambuf_iterator<char>());
    }

    // 讀取 Model (權重)
    string model_content;
    {
        ifstream model_file(MODEL_BIN, ios::binary);
        if (!model_file.is_open()) { cerr << "Model file not found!" << endl; return -1; }
        model_content = string((istreambuf_iterator<char>(model_file)), istreambuf_iterator<char>());
    }

    tnn::ModelConfig model_config;
    model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {proto_content, model_content};

    tnn::TNN tnn_net;
    tnn::Status status = tnn_net.Init(model_config);
    if (status != tnn::TNN_OK) {
        cerr << "TNN Init failed: " << status.description().c_str() << endl;
        return -1;
    }

    // 建立推論實例 (Instance)
    tnn::NetworkConfig network_config;
    network_config.device_type = tnn::DEVICE_ARM; // E9V3 是 ARM 架構
    network_config.enable_tune_kernel = true;    // 關閉 = 加速用啟用
    
    auto instance = tnn_net.CreateInst(network_config, status);
    if (status != tnn::TNN_OK || !instance) {
        cerr << "CreateInst failed: " << status.description().c_str() << endl;
        return -1;
    }
    
    cout << "TNN Model Loaded Successfully!" << endl;

    // ------------------- 3. 攝影機初始化 -------------------
    VideoCapture cap(2); // 根據 Lab3 經驗，確認是 index 2
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera 2" << endl;
        return -1;
    }
    // 嘗試設定解析度以提高讀取速度 (可選)
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, input_blob_mat, display_frame;
    vector<Object> detected_objects;

    cout << "Starting Loop. Press Ctrl+C to stop." << endl;

    while (true) {
        // --- A. 讀取影像 ---
        cap >> frame;
        if (frame.empty()) break;

        // --- B. TNN 預處理 ---
        // 1. Resize 到 224x224
        resize(frame, input_blob_mat, Size(INPUT_WIDTH, INPUT_HEIGHT));
        
        // 2. 轉換格式 (OpenCV BGR -> TNN RGB)
        // TNN MatUtils 可以做，或者我們手動轉
        cvtColor(input_blob_mat, input_blob_mat, COLOR_BGR2RGB);

        // 3. 設定輸入 Blob
        auto input_blob = instance->GetInputBlob(instance->GetAllInputBlobs().begin()->second->GetBlobDesc().name);
        
        // 將 Mat 資料塞入 TNN 輸入
        tnn::MatConvertParam input_cvt_param;
        // 這裡不需要 scale/bias，因為 YOLOv8 模型內部通常已經含有了，或者在 Export 時處理
        // 如果偵測效果很差，可能需要加上 normalization: scale = 1/255.0
        // input_cvt_param.scale = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}; 
        
        // 使用 TNN 提供的工具將 OpenCV Mat 轉成 TNN Blob
        // 注意：這裡假設 input_blob_mat 是連續記憶體
        std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
            tnn::DEVICE_ARM, 
            tnn::N8UC3, 
            std::vector<int>{1, 3, INPUT_HEIGHT, INPUT_WIDTH}, 
            input_blob_mat.data
        );
        
        status = instance->SetInputMat(tnn_mat, input_cvt_param);
        if (status != tnn::TNN_OK) {
            cerr << "SetInputMat Error: " << status.description().c_str() << endl;
            continue;
        }

        // --- C. 執行推論 ---
        status = instance->Forward();
        if (status != tnn::TNN_OK) {
            cerr << "Forward Error: " << status.description().c_str() << endl;
            continue;
        }

        // --- D. 後處理 (解析 + NMS) ---
        postProcess(instance, detected_objects, frame.cols, frame.rows);

        // --- E. 畫圖 (Visualization) ---
        for (const auto& obj : detected_objects) {
            // 畫框
            rectangle(frame, obj.rect, Scalar(0, 255, 0), 2);
            
            // 畫標籤
            string label = CLASS_NAMES[obj.label] + " " + to_string((int)(obj.prob * 100)) + "%";
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            rectangle(frame, Point(obj.rect.x, obj.rect.y - labelSize.height),
                      Point(obj.rect.x + labelSize.width, obj.rect.y + baseLine),
                      Scalar(0, 255, 0), FILLED);
            
            putText(frame, label, Point(obj.rect.x, obj.rect.y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        // --- F. 顯示到 Framebuffer (同 Lab3) ---
        resize(frame, display_frame, Size(DISP_WIDTH, DISP_HEIGHT));
        
        // 關鍵：轉成 BGR565 (嵌入式螢幕常用格式)
        cvtColor(display_frame, display_frame, COLOR_BGR2BGR565);

        // 寫入 mmap 記憶體
        // 假設是全螢幕左上角顯示
        for (int y = 0; y < DISP_HEIGHT; y++) {
            // 計算 framebuffer 的偏移量
            // Lab3 範例: (DISP_Y + y) * width + DISP_X
            long int location = (y * width) * 2; 
            
            // 複製一行 pixels
            memcpy(fbp + location, display_frame.ptr(y), DISP_WIDTH * 2);
        }
        
        // 簡單的時間控制，避免吃滿 CPU
        // waitKey 在這裡沒有視窗效果，但可以稍微 delay
        // cv::waitKey(1); 
    }

    // --- 清理 ---
    munmap(fbp, screensize);
    close(fb);
    cout << "Program exited." << endl;
    return 0;
}