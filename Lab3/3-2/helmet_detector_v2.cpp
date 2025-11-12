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
#include <iterator> // for make_move_iterator

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
    string modelConfig = "yolov3-tiny-helmet.cfg";           // YOLO 設定檔 
    string modelWeights = "yolov3-tiny-helmet_best.weights"; // YOLO 權重檔
    int tileSize = 320;                          // 圖塊大小
    int overlap = 32;                            // 重疊像素
    float confThreshold = 0.2;                   // 信賴度閾值
    float nmsThreshold = 0.3;                    // NMS 閾值
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


// 應用 NMS 過濾重複偵測
vector<Detection> applyNMS(const vector<Detection>& detections,
                           float scoreThreshold,
                           float nmsThreshold) {
    vector<Rect> boxes;
    vector<float> confidences;
    boxes.reserve(detections.size());
    confidences.reserve(detections.size());

    // 準備 NMS 輸入
    for (const auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    // indices 也預留
    vector<int> indices;
    indices.reserve(boxes.size());

    // 執行 NMS（scoreThreshold 是最低分數門檻）
    NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

    // 將過濾結果按 indices 收集回 Detection
    vector<Detection> filteredDetections;
    filteredDetections.reserve(indices.size());
    for (int idx : indices) {
        filteredDetections.push_back(detections[idx]);
    }

    // 日誌：保留的數量（可在正式執行時關掉以減少 I/O）
    cout << "NMS filtering retained " << filteredDetections.size() << " detection results" << endl;

    return filteredDetections;
}

// 對單一圖塊進行偵測
vector<Detection> detectOnTile(Net& net, const Mat& tile, 
                               const Config& config,
                               const vector<String>& outNames) {
    // 準備輸入 blob
    Mat blob;
    blobFromImage(tile, blob, 1/255.0, Size(config.tileSize, config.tileSize), Scalar(), true, false);// true: 用 RGB, false: 不做 crop
    
    net.setInput(blob);
    
    // 前向傳播
    vector<Mat> outputs;
    net.forward(outputs, outNames);
    
    // 提取偵測結果
    // 先 extract
    vector<Detection> tileDetections = extractDetections(outputs, tile.cols, tile.rows, config.confThreshold, config.helmetClassId);
    // 再做 per-tile NMS，避免把過多候選帶到全域合併
    vector<Detection> filtered = applyNMS(tileDetections, config.confThreshold, config.nmsThreshold);
    return filtered;
}

// 主要的分塊偵測函數
vector<Detection> detectWithTiling(Net& net, const Mat& image,
                                   const Config& config,
                                   const vector<String>& outNames) {
    vector<Detection> allDetections;

    int imgHeight = image.rows;
    int imgWidth = image.cols;
    int stride = config.tileSize - config.overlap;

    // 用一致的公式計算估算 tile 數（上限）
    int tilesX = (imgWidth <= config.tileSize) ? 1 : ((imgWidth - config.tileSize) / stride + 1);
    int tilesY = (imgHeight <= config.tileSize) ? 1 : ((imgHeight - config.tileSize) / stride + 1);
    int totalTiles = tilesX * tilesY;

    cout << "start handling tiles, image size: " << imgWidth << "x" << imgHeight << endl;
    cout << "tile size: " << config.tileSize << "x" << config.tileSize
         << ", overlap: " << config.overlap << "px" << endl;
    cout << "estimated number of tiles to process: " << totalTiles << endl;

    // 預留空間：假設每 tile 平均 3~6 個候選（可視情況調整）
    const int AVG_CANDIDATES_PER_TILE = 4;
    allDetections.reserve((size_t)totalTiles * AVG_CANDIDATES_PER_TILE);

    // int tileCount = 0;
    for (int y = 0; y <= imgHeight - config.tileSize; y += stride) {
        for (int x = 0; x <= imgWidth - config.tileSize; x += stride) {
            // tileCount++;

            Rect tileRect(x, y, config.tileSize, config.tileSize);
            Mat tile = image(tileRect); // ROI header, 不會複製像素資料

            // 每個 tile 做偵測（detectOnTile 已包含 per-tile NMS）
            vector<Detection> tileDetections = detectOnTile(net, tile, config, outNames);

            // 把 tile 的本地座標轉為全圖座標
            for (auto& det : tileDetections) {
                det.box.x += x;
                det.box.y += y;
            }

            // 使用 move iterator 批次搬入，避免個別 push_back 的複製
            allDetections.insert(allDetections.end(),
                                 std::make_move_iterator(tileDetections.begin()),
                                 std::make_move_iterator(tileDetections.end()));
            // tileDetections 在此之後會被移動，內容不再可靠

            // // 進度顯示：可改頻率減少 I/O
            // if (tileCount % 20 == 0 || tileCount == totalTiles) {
            //     cout << "processing progress: " << tileCount << "/" << totalTiles
            //          << " (" << (tileCount * 100 / totalTiles) << "%)" << endl;
            // }
        }
    }

    // 處理右邊界與下邊界（如果圖片尺寸不是 stride 的整數倍）
    // 注意：這裡採用 x 與 y 的邊緣起點，並呼叫 detectOnTile 時傳入 outNames
    if ((imgWidth - config.tileSize) % stride != 0 && imgWidth > config.tileSize) {
        int x_edge = imgWidth - config.tileSize;
        // 處理右邊界的所有 tile（包含右下角）
        for (int y = 0; y <= imgHeight - config.tileSize; y += stride) {
            Rect tileRect(x_edge, y, config.tileSize, config.tileSize);
            Mat tile = image(tileRect);

            vector<Detection> tileDetections = detectOnTile(net, tile, config, outNames);
            // 座標轉換
            for (auto& det : tileDetections) {
                det.box.x += x_edge;
                det.box.y += y;
            }
            allDetections.insert(allDetections.end(),
                                 std::make_move_iterator(tileDetections.begin()),
                                 std::make_move_iterator(tileDetections.end()));
        }
    }

    // 處理下邊界（如果圖片高度不是 stride 的整數倍）
    if ((imgHeight - config.tileSize) % stride != 0 && imgHeight > config.tileSize) {
        int y_edge = imgHeight - config.tileSize;
        int x_edge = imgWidth - config.tileSize;
        // 處理下邊界的所有 tile，但排除右下角（已在右邊界處理過）
        for (int x = 0; x <= imgWidth - config.tileSize; x += stride) {
            // ⭐ 關鍵修正：跳過右下角，避免重複處理
            if ((imgWidth - config.tileSize) % stride != 0 && x == x_edge) {
                continue;  // 右下角已在右邊界 loop 處理過
            }
            Rect tileRect(x, y_edge, config.tileSize, config.tileSize);
            Mat tile = image(tileRect);
            vector<Detection> tileDetections = detectOnTile(net, tile, config, outNames);
            // 座標轉換
            for (auto& det : tileDetections) {
                det.box.x += x;
                det.box.y += y_edge;
            }
            allDetections.insert(allDetections.end(),
                                 std::make_move_iterator(tileDetections.begin()),
                                 std::make_move_iterator(tileDetections.end()));
        }
    }

    cout << "tile processing completed, detected " << allDetections.size() << " candidate regions" << endl;

    return allDetections;
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
        cerr << "Usage: " << argv[0] << " <input_image> [output_image]" << endl;
        cerr << "Example: " << argv[0] << " hidden_test_photo.jpg result.jpg" << endl;
        return -1;
    }
    
    string inputPath = argv[1];
    string outputPath = (argc >= 3) ? argv[2] : "lab3_final_24.jpg";
    
    // 初始化設定
    Config config;
    
    // 開始計時
    auto startTime = chrono::high_resolution_clock::now();
    
    cout << "========================================" << endl;
    cout << "  High-Resolution Helmet Detection System" << endl;
    cout << "========================================" << endl;
    
    // 載入 YOLO 模型
    cout << "\n[Step 1] Loading YOLO model..." << endl;
    Net net;
    try {
        net = readNetFromDarknet(config.modelConfig, config.modelWeights);
        
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        cout << "Using CPU backend" << endl;
        
        cout << "Model loaded successfully!" << endl;
    } catch (const Exception& e) {
        cerr << "Error: Unable to load YOLO model" << endl;
        cerr << e.what() << endl;
        return -1;
    }
    
    // 讀取輸入圖片
    cout << "\n[Step 2] Reading input image: " << inputPath << endl;
    Mat image = imread(inputPath);
    
    if (image.empty()) {
        cerr << "Error: Could not read image file " << inputPath << endl;
        return -1;
    }
    
    cout << "Image loaded successfully! Size: " << image.cols << "x" << image.rows << endl;
    
    // 獲取輸出層名稱
    vector<String> outNames = net.getUnconnectedOutLayersNames();

    // 執行分塊偵測
    cout << "\n[Step 3] Running tiled detection..." << endl;
    vector<Detection> allDetections = detectWithTiling(net, image, config, outNames);
    
    // 應用 NMS 過濾
    cout << "\n[Step 4] Applying NMS filtering..." << endl;
    vector<Detection> finalDetections = applyNMS(allDetections, config.confThreshold, config.nmsThreshold);
    
    // 繪製結果
    cout << "\n[Step 5] Drawing detections..." << endl;
    Mat resultImage = image.clone();
    drawDetections(resultImage, finalDetections);
    
    // 儲存結果
    cout << "\n[Step 6] Saving result image: " << outputPath << endl;
    if (imwrite(outputPath, resultImage)) {
        cout << "Result image saved successfully!" << endl;
    } else {
        cerr << "Error: Could not save result image" << endl;
        return -1;
    }
    
    // 計算總執行時間
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
    
    // 輸出最終統計
    cout << "\n========================================" << endl;
    cout << "  Detection completed!" << endl;
    cout << "========================================" << endl;
    cout << "Number of helmets detected: " << finalDetections.size() << endl;
    cout << "Total execution time: " << duration.count() << " seconds" << endl;
    
    if (duration.count() > 1200) {  // 20 分鐘 = 1200 秒
        cout << "Warning: Execution time exceeded 20-minute limit!" << endl;
    }
    
    cout << "========================================" << endl;
    
    return 0;
}