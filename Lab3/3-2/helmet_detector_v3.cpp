/*
 * 多執行緒版本 - 針對 Embedsky E9v3 優化
 * 新增部分:
 * 1. ThreadPool 類別
 * 2. 多執行緒版本的 detectWithTiling()
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iterator>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

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
    int numThreads = 3;  // 新增: 執行緒數量 (4核保留1核給系統)
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

// ============ 新增: Thread Pool 實作 ============

class ThreadPool {
private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    
    mutex queueMutex;
    condition_variable condition;
    atomic<bool> stop;
    atomic<int> activeTasks;
    
public:
    ThreadPool(size_t numThreads) : stop(false), activeTasks(0) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    function<void()> task;
                    
                    {
                        unique_lock<mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        task = move(this->tasks.front());
                        this->tasks.pop();
                    }
                    
                    activeTasks++;
                    task();
                    activeTasks--;
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            unique_lock<mutex> lock(queueMutex);
            if (stop) {
                throw runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace(forward<F>(f));
        }
        condition.notify_one();
    }
    
    void wait() {
        while (true) {
            unique_lock<mutex> lock(queueMutex);
            if (tasks.empty() && activeTasks == 0) {
                break;
            }
            lock.unlock();
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
    
    ~ThreadPool() {
        {
            unique_lock<mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

// ============ 新增: Tile 任務結構 ============

struct TileTask {
    int x, y;           // Tile 在原圖中的位置
    int width, height;  // Tile 尺寸
};

// ============ 修改: 單個 Tile 的偵測函數 (thread-safe 版本) ============

vector<Detection> detectOnTileSafe(Net& net, const Mat& image, 
                                   const TileTask& task,
                                   const Config& config,
                                   const vector<String>& outNames,
                                   mutex& netMutex) {
    // 提取 tile 區域
    Rect tileRect(task.x, task.y, task.width, task.height);
    Mat tile = image(tileRect).clone();  // clone() 確保執行緒安全
    
    // 準備輸入 blob
    Mat blob;
    blobFromImage(tile, blob, 1/255.0, 
                  Size(config.tileSize, config.tileSize), 
                  Scalar(), true, false);
    
    // 使用 mutex 保護 Net 操作 (Net::forward() 不是 thread-safe)
    vector<Mat> outputs;
    {
        lock_guard<mutex> lock(netMutex);
        net.setInput(blob);
        net.forward(outputs, outNames);
    }
    
    // 提取偵測結果
    // 先 extract
    vector<Detection> tileDetections = extractDetections(outputs, tile.cols, tile.rows, config.confThreshold, config.helmetClassId);
    // 再做 per-tile NMS，避免把過多候選帶到全域合併
    vector<Detection> filtered = applyNMS(tileDetections, config.confThreshold, config.nmsThreshold);
    
    // 座標轉換到全圖
    for (auto& det : filtered) {
        det.box.x += task.x;
        det.box.y += task.y;
    }
    
    return filtered;
}

// ============ 新增: 多執行緒版本的 detectWithTiling ============

vector<Detection> detectWithTilingThreaded(Net& net, const Mat& image,
                                           const Config& config,
                                           const vector<String>& outNames) {
    int imgHeight = image.rows;
    int imgWidth = image.cols;
    int stride = config.tileSize - config.overlap;
    
    // 計算所有需要處理的 tiles
    vector<TileTask> tileTasks;
    
    // 主要網格區域
    for (int y = 0; y <= imgHeight - config.tileSize; y += stride) {
        for (int x = 0; x <= imgWidth - config.tileSize; x += stride) {
            tileTasks.push_back({x, y, config.tileSize, config.tileSize});
        }
    }
    
    // 右邊界
    if ((imgWidth - config.tileSize) % stride != 0 && imgWidth > config.tileSize) {
        int x_edge = imgWidth - config.tileSize;
        for (int y = 0; y <= imgHeight - config.tileSize; y += stride) {
            tileTasks.push_back({x_edge, y, config.tileSize, config.tileSize});
        }
    }
    
    // 下邊界 (排除右下角)
    if ((imgHeight - config.tileSize) % stride != 0 && imgHeight > config.tileSize) {
        int y_edge = imgHeight - config.tileSize;
        int x_edge = imgWidth - config.tileSize;
        // 處理下邊界的所有 tile，但排除右下角（已在右邊界處理過）
        for (int x = 0; x <= imgWidth - config.tileSize; x += stride) {
            if ((imgWidth - config.tileSize) % stride != 0 && x == x_edge) {
                continue;  // 跳過右下角
            }
            tileTasks.push_back({x, y_edge, config.tileSize, config.tileSize});
        }
    }
    
    cout << "total tiles to process: " << tileTasks.size() << endl;
    cout << "using " << config.numThreads << " threads" << endl;
    
    // 建立 thread pool
    ThreadPool pool(config.numThreads);
    
    // 共享資源
    vector<Detection> allDetections;
    mutex detectionsMutex;
    mutex netMutex;
    atomic<int> processedTiles(0);
    
    // 提交所有 tile 任務
    for (const auto& task : tileTasks) {
        pool.enqueue([&, task]() {
            // 處理單個 tile
            vector<Detection> tileDetections = detectOnTileSafe(net, image, task, config, outNames, netMutex);
            
            // 將結果加入共享容器
            {
                lock_guard<mutex> lock(detectionsMutex);
                allDetections.insert(allDetections.end(),
                                    make_move_iterator(tileDetections.begin()),
                                    make_move_iterator(tileDetections.end()));
            }
            
            // 更新進度
            int completed = ++processedTiles;
            if (completed % 20 == 0 || completed == (int)tileTasks.size()) {
                cout << "processing progress: " << completed << "/" << tileTasks.size() << " tiles" << endl;
            }
        });
    }
    
    // 等待所有任務完成
    pool.wait();
    
    cout << "tile processing completed, detected " << allDetections.size() << " candidate regions" << endl;
    
    return allDetections;
}

// ============ 修改: 繪製函數 ============

void drawDetections(Mat& image, const vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 繪製矩形框
        rectangle(image, det.box, Scalar(0, 255, 0), 5);
        
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

// ============ 主程式 ============

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_image> [weight] [num_threads]" << endl;
        cerr << "Example: " << argv[0] << " final_demo.jpg yolov3-tiny-helmet_best.weights 3" << endl;
        return -1;
    }
    
    string inputPath = argv[1];
    string outputPath = "lab3_final_24.jpg";

    Config config;
    if (argc >= 3) config.modelWeights = argv[2];
    if (argc >= 4) config.numThreads = atoi(argv[3]);
    
    // 限制執行緒數量 (避免超過硬體核心數)
    if (config.numThreads > 4) {
        cout << "WARNING: Thread count limited to 4" << endl;
        config.numThreads = 4;
    }
    
    auto startTime = chrono::high_resolution_clock::now();
    
    cout << "========================================" << endl;
    cout << "  High-Resolution Helmet Detection System" << endl;
    cout << "========================================" << endl;
    cout << "Thread number: " << config.numThreads << endl;
    
    // 載入 YOLO 模型
    cout << "\n[Step 1] Loading YOLO model..." << endl;
    Net net;
    try {
        net = readNetFromDarknet(config.modelConfig, config.modelWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        cout << "Using CPU backend" << endl;
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

    // 執行多執行緒分塊偵測
    cout << "\n[Step 3] Running multi-thread tiled detection..." << endl;
    vector<Detection> allDetections = detectWithTilingThreaded(net, image, config, outNames);
    
    // 應用全域 NMS 過濾
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
    
    if (duration.count() > 1200) {
        cout << "Warning: Execution time exceeded 20-minute limit!" << endl;
    }
    
    cout << "========================================" << endl;
    
    return 0;
}