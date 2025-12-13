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
    // 確保有輸出
    if (output_blobs.empty()) return;

    tnn::Blob* output_blob = output_blobs.begin()->second;
    float* data = (float*)output_blob->GetHandle().base;
    std::vector<int> shape = output_blob->GetBlobDesc().dims;
    
    // // === DEBUG: 檢查輸出維度 ===
    // // 請在 Terminal 觀察這個輸出，這非常重要！
    // // 正常應該是: [1, 12, 1029] (Channel First) 或 [1, 1029, 12] (Channel Last)
    // cout << "Output Shape: [";
    // for(int d : shape) cout << d << " ";
    // cout << "]" << endl;
    
    // YOLOv8 輸出形狀通常是 [1, 4 + 類別數, Anchors]
    // 也就是 [1, 12, 1029] (針對 224x224 輸入)
    
    // 檢查維度 (YOLOv8 導出的 ONNX 幾乎都是 Channel First: [1, Channels, Anchors])
    int num_channels = shape[1]; // 應該是 4 + 8 = 12
    int num_anchors  = shape[2]; // 224輸入時約為 1029

    // 安全檢查：如果維度反了，交換一下 (雖然 YOLOv8 原生通常是上面那樣)
    bool is_transposed = false; 
    if (num_channels > num_anchors) {
        swap(num_channels, num_anchors);
        is_transposed = true; 
    }

    // 縮放比例：原圖尺寸 / 模型輸入尺寸
    float scale_x = (float)frame_w / INPUT_WIDTH;
    float scale_y = (float)frame_h / INPUT_HEIGHT;

    for (int i = 0; i < num_anchors; ++i) {
        // 1. 找出這個 Anchor 中分數最高的類別
        // YOLOv8 的輸出排列：
        // Channel 0: cx
        // Channel 1: cy
        // Channel 2: w
        // Channel 3: h
        // Channel 4~11: class scores
        
        float max_score = 0.0f;
        int max_label = -1;

        int debug_c;

        // 優化：直接從 Channel 4 開始掃描類別分數
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = 0.0f;
            if (!is_transposed) {
                // data[channel * stride + i]
                score = data[(4 + c) * num_anchors + i];
            } else {
                score = data[i * num_channels + (4 + c)];
            }

            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        // 2. 信心度過濾
        if (max_score > CONF_THRESHOLD) {
            float cx, cy, w, h;
            
            if (!is_transposed) {
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

            // --- 修正：YOLOv8 默認輸出就是 0~224 的像素座標 ---
            // 直接轉回原圖尺寸
            // 左上角座標 x = cx - w/2
            float x = (cx - w * 0.5f) * scale_x;
            float y = (cy - h * 0.5f) * scale_y;
            float width = w * scale_x;
            float height = h * scale_y;

            // 邊界檢查
            int x1 = max(0, min((int)x, frame_w - 1));
            int y1 = max(0, min((int)y, frame_h - 1));
            int w1 = min((int)width, frame_w - x1);
            int h1 = min((int)height, frame_h - y1);

            // // ================= DEBUG 訊息開始 =================
            // // 這裡使用 max_label (不是 c) 和 max_score (不是 score)
            // // 為了不洗版，只印出分數 > 0.5 的 (或是跟上面門檻一致)
            // cout << "[DEBUG] " 
            //     << "Class: " << CLASS_NAMES[max_label] 
            //     << " | Prob: " << fixed << setprecision(2) << max_score 
            //     << " | Raw(TNN): (" << (int)cx << "," << (int)cy << ")" // 模型直接輸出的中心點
            //     << " | Screen: [" << x1 << "," << y1 << " " << w1 << "x" << h1 << "]" // 畫在螢幕上的框
            //     << endl;
            // // ================= DEBUG 訊息結束 =================
            
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
        // 預處理：使用 OpenCV 直接轉成 NCHW Float32
        // 這行會自動完成：Resize(224x224) + SwapRB(BGR->RGB) + Normalize(0~1) + Layout(HWC->NCHW)
        // 這是最穩定的做法，不依賴 TNN 的轉換
        Mat blob;
        cv::dnn::blobFromImage(input_mat, blob, 
                               1.0 / 255.0,             // Scale: 壓縮到 0~1
                               Size(INPUT_WIDTH, INPUT_HEIGHT), // Resize
                               Scalar(0, 0, 0),         // Mean: 0
                               true,                    // SwapRB: BGR -> RGB (重要！)
                               false);                  // Crop

        // 建立 TNN Mat，直接指向 blob 的資料
        // 注意：blob 是連續的 float 記憶體，格式為 NCHW
        std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
            tnn::DEVICE_ARM,
            tnn::NCHW_FLOAT, // 明確指定這是 NCHW 的 Float
            std::vector<int>{1, 3, INPUT_HEIGHT, INPUT_WIDTH}, 
            blob.data // 直接拿 OpenCV 算好的指標
        );

        // 因為我們已經在 blobFromImage 做過 Normalize 了
        // 所以這裡告訴 TNN 不要做任何轉換 (scale=1, bias=0)
        tnn::MatConvertParam input_cvt_param;
        // 預設就是不做轉換，所以其實可以不設，但為了安全：
        input_cvt_param.scale = {1.0, 1.0, 1.0};
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