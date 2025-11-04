/*
 * 嵌入式系統專案 LAB 3 - 程式二
 * 高解析度安全帽偵測系統
 * 
 * 功能: 使用 YOLO 模型在高解析度圖片中偵測安全帽
 * 技術: 圖像分塊 + 座標轉換 + NMS 過濾
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// 偵測結果結構
struct Detection {
    Rect box;           // 邊界框
    float confidence;   // 信賴度
    int classId;        // 類別 ID
};

// 程式設定參數
struct Config {
    string modelConfig = "yolov4-tiny.cfg";      // YOLO 設定檔
    string modelWeights = "yolov4-tiny.weights"; // YOLO 權重檔
    int tileSize = 608;                          // 圖塊大小
    int overlap = 100;                           // 重疊像素
    float confThreshold = 0.5;                   // 信賴度閾值
    float nmsThreshold = 0.4;                    // NMS 閾值
    int helmetClassId = 0;                       // 安全帽類別 ID (需根據模型調整)
};

// 從 YOLO 輸出層提取偵測結果
vector<Detection> extractDetections(const vector<Mat>& outputs, 
                                    int tileWidth, int tileHeight,
                                    float confThreshold, int targetClassId) {
    vector<Detection> detections;
    
    for (const Mat& output : outputs) {
        const float* data = (float*)output.data;
        
        for (int i = 0; i < output.rows; i++) {
            const float* row = data + i * output.cols;
            
            // YOLO 輸出格式: [center_x, center_y, width, height, objectness, class_scores...]
            float objectness = row[4];
            
            if (objectness < confThreshold) continue;
            
            // 找出最高分數的類別
            Mat scores = output.row(i).colRange(5, output.cols);
            Point classIdPoint;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
            
            float confidence = objectness * maxClassScore;
            
            // 只保留目標類別且信賴度足夠的偵測
            if (confidence >= confThreshold && classIdPoint.x == targetClassId) {
                int centerX = (int)(row[0] * tileWidth);
                int centerY = (int)(row[1] * tileHeight);
                int width = (int)(row[2] * tileWidth);
                int height = (int)(row[3] * tileHeight);
                
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                Detection det;
                det.box = Rect(left, top, width, height);
                det.confidence = confidence;
                det.classId = classIdPoint.x;
                
                detections.push_back(det);
            }
        }
    }
    
    return detections;
}

// 對單一圖塊進行偵測
vector<Detection> detectOnTile(Net& net, const Mat& tile, 
                               const Config& config) {
    // 準備輸入 blob
    Mat blob;
    blobFromImage(tile, blob, 1/255.0, Size(config.tileSize, config.tileSize), Scalar(), true, false);
    
    net.setInput(blob);
    
    // 獲取輸出層名稱
    vector<String> outNames = net.getUnconnectedOutLayersNames();
    
    // 前向傳播
    vector<Mat> outputs;
    net.forward(outputs, outNames);
    
    // 提取偵測結果
    return extractDetections(outputs, tile.cols, tile.rows, config.confThreshold, config.helmetClassId);
}

// 主要的分塊偵測函數
vector<Detection> detectWithTiling(Net& net, const Mat& image, 
                                   const Config& config) {
    vector<Detection> allDetections;
    
    int imgHeight = image.rows;
    int imgWidth = image.cols;
    int stride = config.tileSize - config.overlap;
    
    int tileCount = 0;
    int totalTiles = ((imgHeight - config.overlap) / stride + 1) * 
                     ((imgWidth - config.overlap) / stride + 1);
    
    cout << "開始分塊處理，圖片尺寸: " << imgWidth << "x" << imgHeight << endl;
    cout << "圖塊尺寸: " << config.tileSize << "x" << config.tileSize 
         << ", 重疊: " << config.overlap << "px" << endl;
    cout << "預計處理圖塊數: " << totalTiles << endl;
    
    // 雙層迴圈：滑動窗口遍歷整張圖
    for (int y = 0; y <= imgHeight - config.tileSize; y += stride) {
        for (int x = 0; x <= imgWidth - config.tileSize; x += stride) {
            tileCount++;
            
            // 裁切圖塊
            Rect tileRect(x, y, config.tileSize, config.tileSize);
            Mat tile = image(tileRect);
            
            // 對圖塊進行偵測
            vector<Detection> tileDetections = detectOnTile(net, tile, config);
            
            // 將局部座標轉換為全域座標
            for (auto& det : tileDetections) {
                det.box.x += x;
                det.box.y += y;
                allDetections.push_back(det);
            }
            
            // 進度顯示
            if (tileCount % 10 == 0 || tileCount == totalTiles) {
                cout << "處理進度: " << tileCount << "/" << totalTiles 
                     << " (" << (tileCount * 100 / totalTiles) << "%)" << endl;
            }
        }
    }
    
    // 處理邊界（如果圖片尺寸不是 stride 的整數倍）
    // 右邊界
    if (imgWidth % stride != 0) {
        int x = imgWidth - config.tileSize;
        for (int y = 0; y <= imgHeight - config.tileSize; y += stride) {
            Rect tileRect(x, y, config.tileSize, config.tileSize);
            Mat tile = image(tileRect);
            vector<Detection> tileDetections = detectOnTile(net, tile, config);
            
            for (auto& det : tileDetections) {
                det.box.x += x;
                det.box.y += y;
                allDetections.push_back(det);
            }
        }
    }
    
    // 下邊界
    if (imgHeight % stride != 0) {
        int y = imgHeight - config.tileSize;
        for (int x = 0; x <= imgWidth - config.tileSize; x += stride) {
            Rect tileRect(x, y, config.tileSize, config.tileSize);
            Mat tile = image(tileRect);
            vector<Detection> tileDetections = detectOnTile(net, tile, config);
            
            for (auto& det : tileDetections) {
                det.box.x += x;
                det.box.y += y;
                allDetections.push_back(det);
            }
        }
    }
    
    cout << "分塊處理完成，共偵測到 " << allDetections.size() << " 個候選區域" << endl;
    
    return allDetections;
}

// 應用 NMS 過濾重複偵測
vector<Detection> applyNMS(const vector<Detection>& detections, float nmsThreshold) {
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> indices;
    
    // 準備 NMS 輸入
    for (const auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }
    
    // 執行 NMS
    NMSBoxes(boxes, confidences, 0.0, nmsThreshold, indices);
    
    // 收集過濾後的結果
    vector<Detection> filteredDetections;
    for (int idx : indices) {
        filteredDetections.push_back(detections[idx]);
    }
    
    cout << "NMS 過濾後保留 " << filteredDetections.size() << " 個偵測結果" << endl;
    
    return filteredDetections;
}

// 在圖片上繪製偵測結果
void drawDetections(Mat& image, const vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 繪製矩形框
        rectangle(image, det.box, Scalar(0, 255, 0), 2);
        
        // 準備標籤文字
        string label = "Helmet: " + to_string((int)(det.confidence * 100)) + "%";
        
        // 計算文字背景位置
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int top = max(det.box.y, labelSize.height);
        rectangle(image, 
                 Point(det.box.x, top - labelSize.height),
                 Point(det.box.x + labelSize.width, top + baseLine),
                 Scalar(0, 255, 0), FILLED);
        
        // 繪製文字
        putText(image, label, Point(det.box.x, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }
}

int main(int argc, char** argv) {
    // 參數檢查
    if (argc < 2) {
        cerr << "用法: " << argv[0] << " <input_image> [output_image]" << endl;
        cerr << "範例: " << argv[0] << " hidden_test_photo.jpg result.jpg" << endl;
        return -1;
    }
    
    string inputPath = argv[1];
    string outputPath = (argc >= 3) ? argv[2] : "result.jpg";
    
    // 初始化設定
    Config config;
    
    // 開始計時
    auto startTime = chrono::high_resolution_clock::now();
    
    cout << "========================================" << endl;
    cout << "  高解析度安全帽偵測系統" << endl;
    cout << "========================================" << endl;
    
    // 載入 YOLO 模型
    cout << "\n[步驟 1] 載入 YOLO 模型..." << endl;
    Net net;
    try {
        net = readNetFromDarknet(config.modelConfig, config.modelWeights);
        
        // // 設定運算後端（優先使用 OpenCL 或 CUDA）
        // if (cuda::getCudaEnabledDeviceCount() > 0) {
        //     net.setPreferableBackend(DNN_BACKEND_CUDA);
        //     net.setPreferableTarget(DNN_TARGET_CUDA);
        //     cout << "使用 CUDA 加速" << endl;
        // } else {
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            cout << "使用 CPU 運算" << endl;
        // }
        
        cout << "模型載入成功！" << endl;
    } catch (const Exception& e) {
        cerr << "錯誤: 無法載入 YOLO 模型" << endl;
        cerr << e.what() << endl;
        return -1;
    }
    
    // 讀取輸入圖片
    cout << "\n[步驟 2] 讀取輸入圖片: " << inputPath << endl;
    Mat image = imread(inputPath);
    
    if (image.empty()) {
        cerr << "錯誤: 無法讀取圖片檔案 " << inputPath << endl;
        return -1;
    }
    
    cout << "圖片讀取成功！尺寸: " << image.cols << "x" << image.rows << endl;
    
    // 執行分塊偵測
    cout << "\n[步驟 3] 執行分塊偵測..." << endl;
    vector<Detection> allDetections = detectWithTiling(net, image, config);
    
    // 應用 NMS 過濾
    cout << "\n[步驟 4] 應用 NMS 過濾..." << endl;
    vector<Detection> finalDetections = applyNMS(allDetections, config.nmsThreshold);
    
    // 繪製結果
    cout << "\n[步驟 5] 繪製偵測結果..." << endl;
    Mat resultImage = image.clone();
    drawDetections(resultImage, finalDetections);
    
    // 儲存結果
    cout << "\n[步驟 6] 儲存結果圖片: " << outputPath << endl;
    if (imwrite(outputPath, resultImage)) {
        cout << "結果圖片儲存成功！" << endl;
    } else {
        cerr << "錯誤: 無法儲存結果圖片" << endl;
        return -1;
    }
    
    // 計算總執行時間
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
    
    // 輸出最終統計
    cout << "\n========================================" << endl;
    cout << "  偵測完成！" << endl;
    cout << "========================================" << endl;
    cout << "偵測到的安全帽數量: " << finalDetections.size() << endl;
    cout << "總執行時間: " << duration.count() << " 秒" << endl;
    
    if (duration.count() > 1200) {  // 20 分鐘 = 1200 秒
        cout << "警告: 執行時間超過 20 分鐘限制！" << endl;
    }
    
    cout << "========================================" << endl;
    
    return 0;
}