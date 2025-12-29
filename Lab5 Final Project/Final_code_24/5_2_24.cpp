//320 * 320 
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/blob_converter.h"
using namespace std;
using namespace cv;

// ================= 模型設定 (配合 YOLOv8 320) =================
const int INPUT_WIDTH = 320;
const int INPUT_HEIGHT = 320;

// ================= 結構定義 =================
struct Object {
    cv::Rect rect;
    int label;
    float prob;
};

struct AppConfig {
    string img_path;
    string model_proto;
    string model_bin;
    string classes_path; // 新增：類別檔路徑
    int tile_size;
    int overlap;
    float conf_thres;
    float nms_thres;
};

// ================= 工具函式 =================

// 讀取 classes.txt
vector<string> load_classes(const string& path) {
    vector<string> classes;
    ifstream file(path);
    string line;
    if (!file.is_open()) {
        cerr << "Error: Could not open classes file: " << path << endl;
        return classes;
    }
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // 去除 Windows 換行
        classes.push_back(line);
    }
    return classes;
}

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

void nms(vector<Object>& objects, float iou_threshold) {
    if (objects.empty()) return;
    sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    vector<bool> is_suppressed(objects.size(), false);
    for (size_t i = 0; i < objects.size(); ++i) {
        if (is_suppressed[i]) continue;
        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (is_suppressed[j]) continue;
            if (objects[i].label != objects[j].label) continue;
            if (get_iou(objects[i].rect, objects[j].rect) > iou_threshold) {
                is_suppressed[j] = true;
            }
        }
    }
    vector<Object> kept;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (!is_suppressed[i]) kept.push_back(objects[i]);
    }
    objects = kept;
}

// ================= TNN 推論核心 (NCHW + 診斷修正版) =================
// ================= 最終修正版：BlobConverter + NCHW =================
// ================= 最終完美版：Letterbox + BlobConverter + NCHW =================
// ================= 最終修復版：去除雙重 Sigmoid + Letterbox =================
vector<Object> inference_one_tile(std::shared_ptr<tnn::Instance> instance, const Mat& tile_img, float conf_threshold, int num_classes) {
    vector<Object> objects;

    // 1. Letterbox 預處理 (保持比例，補黑邊，解決準度問題)
    int target_w = INPUT_WIDTH;  // 640
    int target_h = INPUT_HEIGHT; // 640
    Mat canvas(target_h, target_w, CV_8UC3, Scalar(114, 114, 114));
    float ratio = min((float)target_w / tile_img.cols, (float)target_h / tile_img.rows);
    int new_unpad_w = (int)(tile_img.cols * ratio);
    int new_unpad_h = (int)(tile_img.rows * ratio);
    Mat resized_img;
    if (tile_img.cols != new_unpad_w || tile_img.rows != new_unpad_h) {
        resize(tile_img, resized_img, Size(new_unpad_w, new_unpad_h));
    } else {
        resized_img = tile_img;
    }
    int dw = (target_w - new_unpad_w) / 2;
    int dh = (target_h - new_unpad_h) / 2;
    resized_img.copyTo(canvas(Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // 2. 轉換為 TNN 輸入
    Mat blob;
    cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, Size(), Scalar(0, 0, 0), true, false);

    std::shared_ptr<tnn::Mat> tnn_mat = std::make_shared<tnn::Mat>(
        tnn::DEVICE_ARM, tnn::NCHW_FLOAT, vector<int>{1, 3, INPUT_HEIGHT, INPUT_WIDTH}, blob.data
    );
    instance->SetInputMat(tnn_mat, tnn::MatConvertParam()); 
    instance->Forward();

    // 3. 取得並轉換輸出 (BlobConverter)
    tnn::BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);
    tnn::Blob* output_blob = output_blobs.begin()->second;
    tnn::BlobConverter converter(output_blob);
    tnn::Mat cpu_mat(tnn::DEVICE_ARM, tnn::NCHW_FLOAT, output_blob->GetBlobDesc().dims);
    tnn::Status status = converter.ConvertToMat(cpu_mat, tnn::MatConvertParam(), NULL);
    if (status != tnn::TNN_OK) return objects;

    float* data = (float*)cpu_mat.GetData();
    
    // 4. 解析輸出 (修正：直接讀取機率，不做 Sigmoid)
    int num_anchors = 2100; 
    int stride = 2100; 

    // Debug: 印出前幾個數值確認 (只印一次)
    static bool printed = false;
    if (!printed) {
        float debug_score = data[4 * stride + 0]; // 第一個 anchor 的第一個類別分數
        cout << "DEBUG: Raw Score Sample = " << debug_score << endl;
        if (debug_score > 0.0f && debug_score < 1.0f) {
            cout << "DEBUG: Value is already 0~1 (Probability). No Sigmoid needed." << endl;
        } else {
            cout << "DEBUG: Value looks like Logit. Sigmoid MIGHT be needed." << endl;
        }
        printed = true;
    }

    for (int i = 0; i < num_anchors; ++i) {
        float cx = data[0 * stride + i];
        float cy = data[1 * stride + i];
        float w  = data[2 * stride + i];
        float h  = data[3 * stride + i];
        
        if (w < 1.0f || h < 1.0f) continue;

        float max_score = 0.0f; // 這裡直接存機率，初始化為 0
        int max_label = -1;

        for (int c = 0; c < num_classes; ++c) {
            float val = data[(4 + c) * stride + i];
            // 【修正】直接拿原始值當機率
            if (val > max_score) {
                max_score = val;
                max_label = c;
            }
        }

        // 這裡不需要 prob = sigmoid(max_score) 了！
        float prob = max_score; 

        if (prob > conf_threshold) {
            // 還原 Letterbox 座標
            float x_canvas = cx - w / 2.0f;
            float y_canvas = cy - h / 2.0f;
            float x_unpad = x_canvas - dw;
            float y_unpad = y_canvas - dh;
            float x = x_unpad / ratio;
            float y = y_unpad / ratio;
            float final_w = w / ratio;
            float final_h = h / ratio;

            if (x < 0) x = 0;
            if (y < 0) y = 0;

            Object obj;
            obj.rect = Rect((int)x, (int)y, (int)final_w, (int)final_h);
            obj.label = max_label;
            obj.prob = prob;
            objects.push_back(obj);
        }
    }
    
    nms(objects, 0.45f);
    return objects;
}

// ================= 主程式 =================
int main(int argc, char** argv) {
    // 1. 參數設定
    const String keys =
        "{help h usage ? |      | print this message    }"
        "{@image         |      | (required) input image path }"
        "{proto p        |best.tnnproto| model proto path }"
        "{model m        |best.tnnmodel| model bin path   }"
        "{classes c      |classes.txt  | classes file path}"
        "{tile t         |320   | tile size (default 320 for YOLOv8) }"
        "{overlap o      |80   | tile overlap size       }"
        "{conf           |0.25  | confidence threshold    }"
        "{nms            |0.45  | global nms iou threshold }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("E9V3 YOLOv8 Detector");

    if (parser.has("help") || !parser.has("@image")) {
        parser.printMessage();
        return 0;
    }

    AppConfig cfg;
    cfg.img_path = parser.get<string>("@image");
    cfg.model_proto = parser.get<string>("proto");
    cfg.model_bin = parser.get<string>("model");
    cfg.classes_path = parser.get<string>("classes");
    cfg.tile_size = parser.get<int>("tile");
    cfg.overlap = parser.get<int>("overlap");
    cfg.conf_thres = parser.get<float>("conf");
    cfg.nms_thres = parser.get<float>("nms");

    // 2. 載入類別檔
    vector<string> class_names = load_classes(cfg.classes_path);
    if (class_names.empty()) {
        cerr << "Error: No classes loaded from " << cfg.classes_path << endl;
        return -1;
    }
    cout << "Loaded " << class_names.size() << " classes." << endl;

    // 3. 初始化 TNN
    string proto_content, model_content;
    {
        ifstream pf(cfg.model_proto);
        ifstream mf(cfg.model_bin, ios::binary);
        if (!pf || !mf) { cerr << "Error: TNN Model files not found!" << endl; return -1; }
        proto_content = string((istreambuf_iterator<char>(pf)), istreambuf_iterator<char>());
        model_content = string((istreambuf_iterator<char>(mf)), istreambuf_iterator<char>());
    }

    tnn::ModelConfig model_config;
    model_config.model_type = tnn::MODEL_TYPE_TNN;
    model_config.params = {proto_content, model_content};
    tnn::TNN tnn_net;
    if (tnn_net.Init(model_config) != tnn::TNN_OK) { cerr << "TNN Init failed" << endl; return -1; }

    tnn::NetworkConfig network_config;
    network_config.device_type = tnn::DEVICE_ARM; 

    // // 👇 新增這一行：允許低精度運算 (這對 INT8 是必須的)
    // network_config.precision = tnn::PRECISION_LOW;


    tnn::Status status;
    auto instance = tnn_net.CreateInst(network_config, status);

    // 4. 讀圖與切圖邏輯 (Tiling)
    Mat big_img = imread(cfg.img_path);
    if (big_img.empty()) { cerr << "Error: Cannot open image" << endl; return -1; }
    
    int img_w = big_img.cols;
    int img_h = big_img.rows;
    vector<Object> global_objects;

    // 如果圖片比 Tile 還小，直接把 Tile 設為圖片大小
    if (img_w < cfg.tile_size) cfg.tile_size = img_w;
    if (img_h < cfg.tile_size) cfg.tile_size = img_h;

    int step = cfg.tile_size - cfg.overlap; 
    if (step <= 0) step = cfg.tile_size; // 防呆

    cout << "Processing image (" << img_w << "x" << img_h << ") with tile size " << cfg.tile_size << "..." << endl;

    for (int y = 0; y < img_h; y += step) {
        for (int x = 0; x < img_w; x += step) {
            // 邊界處理
            int w = cfg.tile_size;
            int h = cfg.tile_size;
            
            // 如果超過邊界，往回退一點，確保 Crop 始終是 tile_size 大小 (除非圖本身就太小)
            int start_x = x;
            int start_y = y;
            
            if (start_x + w > img_w) start_x = max(0, img_w - w);
            if (start_y + h > img_h) start_y = max(0, img_h - h);
            
            // 真正的 ROI 區域
            Rect roi(start_x, start_y, min(w, img_w - start_x), min(h, img_h - start_y));
            Mat tile = big_img(roi).clone();

            // 執行推論
            vector<Object> tile_objs = inference_one_tile(instance, tile, cfg.conf_thres, class_names.size());

            // 座標還原回大圖
            for (auto& obj : tile_objs) {
                obj.rect.x += start_x;
                obj.rect.y += start_y;
                
                // 邊界限制
                obj.rect.x = max(0, obj.rect.x);
                obj.rect.y = max(0, obj.rect.y);
                if (obj.rect.x + obj.rect.width > img_w) obj.rect.width = img_w - obj.rect.x;
                if (obj.rect.y + obj.rect.height > img_h) obj.rect.height = img_h - obj.rect.y;
                
                global_objects.push_back(obj);
            }

            // 避免無限迴圈 (當圖小於 step 時)
            if (start_x + w >= img_w && start_y + h >= img_h) goto end_tiling; 
        }
    }
    end_tiling:;

    // 5. 全域 NMS
    nms(global_objects, cfg.nms_thres);
    cout << "Found " << global_objects.size() << " objects." << endl;

    // 6. 繪圖 (修正文字跑出螢幕問題)
    for (const auto& obj : global_objects) {
        Scalar color(0, 255, 0); // Green
        int thickness = 2;
        rectangle(big_img, obj.rect, color, thickness);
        
        string label_text = (obj.label < class_names.size() ? class_names[obj.label] : to_string(obj.label));
        string text = label_text + " " + to_string((int)(obj.prob * 100)) + "%";
        
        // --- 🔥 修改開始：智慧調整文字位置 ---
        
        // 1. 先計算文字的高度
        int baseLine = 0;
        Size labelSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        
        // 2. 預設位置：在框框上方
        int text_y = obj.rect.y - 5;
        
        // 3. 檢查：如果上方空間小於文字高度 (代表會切到)，就改寫在框框內部
        if (text_y < labelSize.height) {
            text_y = obj.rect.y + labelSize.height + 5;
        }
        
        putText(big_img, text, Point(obj.rect.x, text_y), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        // --- 🔥 修改結束 ---
    }

    // 儲存結果
    imwrite("result.jpg", big_img);
    cout << "Saved to result.jpg" << endl;

    // 7. Framebuffer 顯示 (E9V3 專用 - 修正版)
    int fb = open("/dev/fb0", O_RDWR);
    if (fb >= 0) {
        struct fb_var_screeninfo vinfo;
        struct fb_fix_screeninfo finfo; // 1. 新增這個結構宣告

        // 2. 取得 variable info (解析度等)
        ioctl(fb, FBIOGET_VSCREENINFO, &vinfo);
        
        // 3. 取得 fixed info (關鍵：為了拿到 line_length)
        ioctl(fb, FBIOGET_FSCREENINFO, &finfo);

        int sw = vinfo.xres; 
        int sh = vinfo.yres; 
        int bpp = vinfo.bits_per_pixel;
        
        // Mmap
        long screensize = sw * sh * bpp / 8;
        unsigned char* fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
        
        if (fbp != MAP_FAILED) {
            Mat disp;
            float r = min((float)sw/img_w, (float)sh/img_h);
            resize(big_img, disp, Size(), r, r); // 等比例縮放以適應螢幕
            
            // 居中顯示
            int start_x = (sw - disp.cols) / 2;
            int start_y = (sh - disp.rows) / 2;

            if (bpp == 32) {
                // E9V3 通常是 BGRA (32bit)
                Mat disp_bgra;
                cvtColor(disp, disp_bgra, COLOR_BGR2BGRA);
                
                for(int y = 0; y < disp.rows; y++) {
                    if (y + start_y >= sh) break;
                    
                    // 修正：使用 finfo.line_length 來計算記憶體偏移量
                    long location = (start_x + vinfo.xoffset) * (bpp/8) + 
                                    (y + start_y + vinfo.yoffset) * finfo.line_length;
                                    
                    memcpy(fbp + location, disp_bgra.ptr(y), disp.cols * 4);
                }
            } else if (bpp == 16) {
                // RGB565
                Mat disp_565;
                cvtColor(disp, disp_565, COLOR_BGR2BGR565);
                for(int y = 0; y < disp.rows; y++) {
                    if (y + start_y >= sh) break;
                    
                    // 修正：使用 finfo.line_length
                    long location = (start_x + vinfo.xoffset) * (bpp/8) + 
                                    (y + start_y + vinfo.yoffset) * finfo.line_length;
                                    
                    memcpy(fbp + location, disp_565.ptr(y), disp.cols * 2);
                }
            }
            munmap(fbp, screensize);
        }
        close(fb);
    } else {
        cout << "Warning: Cannot open framebuffer /dev/fb0" << endl;
    }

    return 0;
}