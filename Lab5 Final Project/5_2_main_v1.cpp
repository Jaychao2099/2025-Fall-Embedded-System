#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <fstream>
#include <condition_variable>
#include <queue>

#include <opencv2/opencv.hpp>
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/mat_utils.h"
#include <arm_neon.h>

using namespace std;
using namespace cv;

// ================= 參數設定區 =================
const string MODEL_PROTO = "./yolov8n_416.tnnproto"; 
const string MODEL_BIN   = "./yolov8n_416.tnnmodel";

// 針對 Photo Recognition 的設定
const int INPUT_SIZE = 416;        // 模型輸入大小 (建議 416 or 320)
const int TILE_SIZE  = 416;        // 切圖大小 (通常等於輸入大小，1:1 不縮放)
const int OVERLAP    = 125;        // 重疊像素 (約 30%)
const float CONF_THRESHOLD = 0.35; // 靜態偵測門檻可稍低
const float NMS_THRESHOLD  = 0.35; // NMS 門檻
const int NUM_THREADS = 3;         // 保留一核給主控

// 類別名稱 (請確認順序一致)
const vector<string> CLASS_NAMES = {
    "Airplane",
    "Apple",
    "banana",
    "Baseball",
    "book",
    "Bottle",
    "Cat",
    "cell phone",
    "Controller",
    "cup",
    "Dart",
    "Diningtable",
    "Dolphin",
    "Drink",
    "Fork",
    "Glass",
    "Glasses",
    "Hammer",
    "Keyboard",
    "Knife",
    "Monitor",
    "Motorcycle",
    "Mouse",
    "Mug",
    "Orange",
    "Pen",
    "Pencil",
    "Pigeon",
    "Pizza",
    "Poker card",
    "Refrigerator",
    "Ruler",
    "Scissors",
    "spoon",
    "Sticky note",
    "Tennis",
    "Tissue",
    "Umbrella",
    "USB drive",
    "Zebra",
};

// ================= 資料結構 =================
struct Object {
    cv::Rect rect;
    int label;
    float prob;
};

struct TileTask {
    int id;
    int x; // 在 Pad 後大圖上的 x 座標
    int y; // 在 Pad 後大圖上的 y 座標
};

// ================= 全域變數 =================
// 讀入記憶體後的模型內容 (避免每個 Thread 都去讀檔)
string g_proto_content;
string g_model_content;

// 待處理任務佇列
queue<TileTask> g_task_queue;
mutex g_queue_mutex;

// 存放所有偵測結果
vector<Object> g_all_detections;
mutex g_result_mutex;

// ================= NEON NMS (沿用作業一) =================
// 為了節省篇幅，這裡保留核心邏輯
void nms_neon(vector<Object>& objects, float threshold) {
    if (objects.empty()) return;
    
    // 1. Sort by probability descending
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

    // Padding for NEON
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
            
            // IoU Check
            float32x4_t limit = vmulq_f32(union_area, v_thresh);
            uint32x4_t mask = vcgtq_f32(inter_area, limit);

            uint32_t res[4];
            vst1q_u32(res, mask);

            for (int k = 0; k < 4; ++k) {
                int cur_idx = j + k;
                if (cur_idx < n && res[k] != 0) {
                    objects[cur_idx].prob = 0; // Mark as removed
                }
            }
        }
    }
    objects.erase(remove_if(objects.begin(), objects.end(), 
        [](const Object& obj) { return obj.prob == 0; }), objects.end());
}

// // 輔助函式：計算 NC4HW4 格式的記憶體偏移量
// // block_size 在 ARM 上通常是 4
// inline int get_nc4hw4_index(int anchor_idx, int channel_idx, int area) {
//     int block = channel_idx / 4;
//     int remain = channel_idx % 4;
//     // 公式：Block偏移 + Anchor偏移 + Channel內偏移
//     return block * (area * 4) + anchor_idx * 4 + remain;
// }

// ================= YOLOv8 PostProcess =================
// 解析單個 Tile 的輸出
void postProcessTile(std::shared_ptr<tnn::Instance> instance, 
                     vector<Object>& objects, 
                     int tile_off_x, int tile_off_y) {
    
    tnn::BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);
    if (output_blobs.empty()) return;

    tnn::Blob* output_blob = output_blobs.begin()->second;
    float* data = (float*)output_blob->GetHandle().base;
    
    // YOLOv8n 416x416 的 Anchor 數量 (需根據模型實際輸出調整)
    // 416/8=52, 416/16=26, 416/32=13 -> 52*52 + 26*26 + 13*13 = 2704 + 676 + 169 = 3549
    int num_anchors = 3549;
    int num_classes = 40;
    
    // 注意：TNN 輸出通常是 NC4HW4 或 NCHW，需根據你的轉換設定
    // 這裡假設我們在 Worker 裡使用了與 Task1 相同的存取邏輯 (helper function)
    // 為了簡化，這裡直接寫最通用的提取邏輯
    
    auto get_index = [&](int anchor, int channel) {
        // 簡易公式，實際建議使用 Task 1 的 get_nc4hw4_index
        // 這裡假設輸出已經被轉成 float array，格式需驗證
        // 若使用 TNN 預設，請複製 Task 1 的 get_nc4hw4_index 函式進來
        int block = channel / 4;
        int remain = channel % 4;
        return block * (num_anchors * 4) + anchor * 4 + remain;
    };

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = 0.0f;
        int max_label = -1;

        // 找最高分類
        for (int c = 0; c < num_classes; ++c) {
            float score = data[get_index(i, 4 + c)]; 
            if (score > max_score) {
                max_score = score;
                max_label = c;
            }
        }

        if (max_score > CONF_THRESHOLD) {
            float cx = data[get_index(i, 0)];
            float cy = data[get_index(i, 1)];
            float w  = data[get_index(i, 2)];
            float h  = data[get_index(i, 3)];

            // 轉回 Tile 座標 (因為輸入就是 416，所以 scale = 1)
            float x = (cx - w * 0.5f);
            float y = (cy - h * 0.5f);

            // 重要：加上 Tile 在大圖的偏移量
            Object obj;
            obj.rect = cv::Rect(
                (int)(x + tile_off_x), 
                (int)(y + tile_off_y), 
                (int)w, (int)h
            );
            obj.label = max_label;
            obj.prob = max_score;
            objects.push_back(obj);
        }
    }
}

// ================= Worker 執行緒 =================
void worker_thread_func(int thread_id, const Mat& padded_img) {
    // 1. 初始化 TNN (每個執行緒獨立一份)
    tnn::ModelConfig model_config;
    model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {g_proto_content, g_model_content};
    
    tnn::TNN tnn_net;
    if (tnn_net.Init(model_config) != tnn::TNN_OK) {
        cerr << "[Thread " << thread_id << "] Init TNN Failed!" << endl;
        return;
    }

    tnn::NetworkConfig network_config;
    network_config.device_type = tnn::DEVICE_ARM;
    tnn::Status status;
    auto instance = tnn_net.CreateInst(network_config, status);
    
    if (status != tnn::TNN_OK) return;

    // 2. 循環接任務
    while (true) {
        TileTask task;
        bool has_task = false;
        {
            lock_guard<mutex> lock(g_queue_mutex);
            if (!g_task_queue.empty()) {
                task = g_task_queue.front();
                g_task_queue.pop();
                has_task = true;
            }
        }

        if (!has_task) break; // 沒任務就結束

        // 3. 執行推論
        // 從大圖切出 ROI (唯讀操作，Thread Safe)
        cv::Rect roi(task.x, task.y, TILE_SIZE, TILE_SIZE);
        Mat tile = padded_img(roi).clone(); // Clone 出來變成連續記憶體

        // Preprocess
        Mat blob;
        cv::dnn::blobFromImage(tile, blob, 1.0/255.0, Size(INPUT_SIZE, INPUT_SIZE), Scalar(0,0,0), true, false);

        std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
            tnn::DEVICE_ARM, tnn::NCHW_FLOAT, vector<int>{1, 3, INPUT_SIZE, INPUT_SIZE}, blob.data
        );
        
        instance->SetInputMat(tnn_mat, tnn::MatConvertParam());
        instance->Forward();

        // 4. Postprocess
        vector<Object> local_objs;
        // 這裡需要把 Task 1 的 get_nc4hw4_index 邏輯帶入
        // 為節省篇幅，假設 postProcessTile 已經實作完畢
        postProcessTile(instance, local_objs, task.x, task.y);

        // 5. 存回全域結果
        if (!local_objs.empty()) {
            lock_guard<mutex> lock(g_result_mutex);
            g_all_detections.insert(g_all_detections.end(), local_objs.begin(), local_objs.end());
        }

        // printf("[Thread %d] Processed Tile (%d, %d)\n", thread_id, task.x, task.y);
    }
}

// ================= 工具：讀取檔案到字串 =================
bool loadFile(const string& path, string& content) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) return false;
    content = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    return true;
}

// ================= 主程式 =================
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    string img_path = argv[1];
    
    // 1. 載入模型檔案
    cout << "Loading model..." << endl;
    if (!loadFile(MODEL_PROTO, g_proto_content) || !loadFile(MODEL_BIN, g_model_content)) {
        cerr << "Error: Cannot load model files." << endl;
        return -1;
    }

    // 2. 讀取並 Padding 圖片
    cout << "Reading image: " << img_path << endl;
    Mat raw_img = imread(img_path);
    if (raw_img.empty()) {
        cerr << "Error: Image not found." << endl;
        return -1;
    }

    // === 實作你的「完美 Padding」邏輯 ===
    int stride = TILE_SIZE - OVERLAP;
    
    // 計算水平需要幾個 Tile
    // 公式： (Width - TileSize) / Stride + 1，取 Ceil
    // 但更直觀的是：我們從 0 開始，每次加 stride，直到覆蓋原圖寬度
    int tiles_x = 0;
    int current_w = 0;
    while (current_w < raw_img.cols) {
        tiles_x++;
        if (current_w + TILE_SIZE >= raw_img.cols) break;
        current_w += stride;
    }
    
    int tiles_y = 0;
    int current_h = 0;
    while (current_h < raw_img.rows) {
        tiles_y++;
        if (current_h + TILE_SIZE >= raw_img.rows) break;
        current_h += stride;
    }

    // 計算需要的總寬高：最後一個 Tile 的起始點 + TileSize
    int target_w = (tiles_x - 1) * stride + TILE_SIZE;
    int target_h = (tiles_y - 1) * stride + TILE_SIZE;
    
    // 處理特例：如果原圖比 Tile 還小
    if (raw_img.cols < TILE_SIZE) target_w = TILE_SIZE;
    if (raw_img.rows < TILE_SIZE) target_h = TILE_SIZE;

    int pad_right = target_w - raw_img.cols;
    int pad_bottom = target_h - raw_img.rows;

    Mat padded_img;
    // 使用黑色 (0,0,0) 填充邊緣
    copyMakeBorder(raw_img, padded_img, 0, pad_bottom, 0, pad_right, BORDER_CONSTANT, Scalar(0,0,0));

    cout << "Original Size: " << raw_img.cols << "x" << raw_img.rows << endl;
    cout << "Padded Size  : " << padded_img.cols << "x" << padded_img.rows << endl;
    cout << "Grid Layout  : " << tiles_x << " x " << tiles_y << " = " << tiles_x * tiles_y << " tiles." << endl;

    // 3. 產生任務列表
    int task_id = 0;
    for (int y = 0; y < padded_img.rows; y += stride) {
        for (int x = 0; x < padded_img.cols; x += stride) {
            // 安全檢查：不要超出邊界 (雖然照理說不會)
            if (x + TILE_SIZE > padded_img.cols || y + TILE_SIZE > padded_img.rows) continue;
            
            TileTask task;
            task.id = task_id++;
            task.x = x;
            task.y = y;
            g_task_queue.push(task);
        }
    }

    // 4. 啟動多執行緒
    cout << "Starting " << NUM_THREADS << " worker threads..." << endl;
    auto start_time = chrono::high_resolution_clock::now();

    vector<thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        // 注意：傳遞 padded_img 參考
        threads.emplace_back(worker_thread_func, i, std::ref(padded_img));
    }

    // 等待所有執行緒結束
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Inference finished in " << duration / 1000.0 << " seconds." << endl;
    cout << "Total Raw Detections: " << g_all_detections.size() << endl;

    // 5. 全域 NMS (去除 Tiling 重疊造成的重複框)
    nms_neon(g_all_detections, NMS_THRESHOLD);
    cout << "After NMS: " << g_all_detections.size() << " objects." << endl;

    // 6. 畫圖並存檔 (畫在原圖上，超過原圖範圍的框框做裁切)
    for (const auto& obj : g_all_detections) {
        // 限制框框不要畫出原圖範圍
        Rect safe_rect = obj.rect & Rect(0, 0, raw_img.cols, raw_img.rows);
        
        if (safe_rect.area() > 0) {
            rectangle(raw_img, safe_rect, Scalar(0, 0, 255), 2); // 紅色框
            string label = CLASS_NAMES[obj.label]; // + " " + to_string((int)(obj.prob*100));
            
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
            int text_y = safe_rect.y - 5;
            if (text_y < labelSize.height) text_y = safe_rect.y + labelSize.height + 5;

            rectangle(raw_img, Point(safe_rect.x, text_y - labelSize.height),
                      Point(safe_rect.x + labelSize.width, text_y + baseLine),
                      Scalar(0, 0, 255), FILLED);
            putText(raw_img, label, Point(safe_rect.x, text_y),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        }
    }

    string out_path = "result_2_2.jpg";
    imwrite(out_path, raw_img);
    cout << "Result saved to " << out_path << endl;

    return 0;
}