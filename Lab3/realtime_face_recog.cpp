#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <map>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;

// 定義 ID 到學號的對應表（ID 編號必須與訓練資料集的目錄順序一致！）
map<int, string> createIdToStudentIdMap() {
    map<int, string> idMap;
    idMap[0] = "314552021";  // user_0 對應的學號
    idMap[1] = "314554053";  // user_1 對應的學號
    // 根據您的組員數量繼續新增...
    return idMap;
}

int main(int argc, char** argv) {
    // === FR-1.1: 系統初始化 ===
    
    // 1. 載入 Haar Cascade 人臉偵測器
    string cascadePath = "haarcascade_frontalface_default.xml";
    CascadeClassifier faceCascade;
    
    if (!faceCascade.load(cascadePath)) {
        cerr << "錯誤: 無法載入人臉偵測模型 " << cascadePath << endl;
        return -1;
    }
    cout << "✓ 成功載入人臉偵測模型" << endl;
    
    // 2. 載入 LBPH 人臉辨識模型
    string modelPath = "lbph_model.yml";
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    
    try {
        recognizer->read(modelPath);
        cout << "✓ 成功載入人臉辨識模型" << endl;
    } catch (Exception& e) {
        cerr << "錯誤: 無法載入人臉辨識模型 " << modelPath << endl;
        cerr << "詳細資訊: " << e.what() << endl;
        return -1;
    }
    
    // 3. 初始化攝影機
    VideoCapture camera(2);  // 2 號攝影機
    
    if (!camera.isOpened()) {
        cerr << "錯誤: 無法開啟攝影機" << endl;
        return -1;
    }
    cout << "✓ 成功開啟攝影機" << endl;
    
    // 建立 ID 對應表
    map<int, string> idToStudentId = createIdToStudentIdMap();
    
    // 效能優化參數
    const Size TARGET_SIZE(640, 480);  // 目標解析度
    const int FRAME_SKIP = 2;          // 每 3 幀處理一次（跳過 2 幀）
    const double CONFIDENCE_THRESHOLD = 100.0;  // 信賴度門檻（LBPH 中越低越好）
    
    int frameCount = 0;
    vector<Rect> lastFaces;  // 儲存上一次偵測到的人臉位置
    vector<string> lastLabels;  // 儲存上一次的辨識結果
    
    cout << "\n系統已啟動，按 'q' 或 ESC 退出...\n" << endl;
    
    // === 主迴圈 ===
    while (true) {
        // === FR-1.2: 影像擷取與預處理 ===
        Mat frame, resizedFrame, grayFrame;
        
        // 讀取一幀影像
        camera >> frame;
        if (frame.empty()) {
            cerr << "警告: 無法讀取影像幀" << endl;
            break;
        }
        
        // 縮放影像以提升效能
        resize(frame, resizedFrame, TARGET_SIZE);
        
        // 轉換為灰階（用於偵測與辨識）
        cvtColor(resizedFrame, grayFrame, COLOR_BGR2GRAY);
        
        // 跳幀處理以提升即時性
        bool shouldProcess = (frameCount % (FRAME_SKIP + 1) == 0);
        
        if (shouldProcess) {
            // === FR-1.3: 人臉偵測 ===
            vector<Rect> faces;
            faceCascade.detectMultiScale(
                grayFrame,
                faces,
                1.1,        // scaleFactor
                5,          // minNeighbors
                0,          // flags
                Size(30, 30) // minSize
            );
            
            // === FR-1.4: 人臉辨識 ===
            vector<string> labels;
            
            for (size_t i = 0; i < faces.size(); i++) {
                // 裁切人臉 ROI
                Mat faceROI = grayFrame(faces[i]);
                
                // 進行辨識
                int predictedLabel = -1;
                double confidence = 0.0;
                recognizer->predict(faceROI, predictedLabel, confidence);
                
                // 根據信賴度判斷結果
                string label;
                if (confidence < CONFIDENCE_THRESHOLD && 
                    idToStudentId.find(predictedLabel) != idToStudentId.end()) {
                    label = idToStudentId[predictedLabel];
                } else {
                    label = "unknown";
                }
                
                // 附加信賴度資訊（可選，用於除錯）
                label += " (" + to_string((int)confidence) + ")";
                
                labels.push_back(label);
            }
            
            // 更新最後的偵測結果
            lastFaces = faces;
            lastLabels = labels;
        }
        
        // === FR-1.5: 結果視覺化 ===
        // 在原始彩色影像上繪製結果
        for (size_t i = 0; i < lastFaces.size() && i < lastLabels.size(); i++) {
            // 繪製矩形框
            rectangle(resizedFrame, lastFaces[i], Scalar(0, 255, 0), 2);
            
            // 繪製文字標籤
            int baseline = 0;
            Size textSize = getTextSize(
                lastLabels[i], 
                FONT_HERSHEY_SIMPLEX, 
                0.6, 
                2, 
                &baseline
            );
            
            Point textOrg(
                lastFaces[i].x,
                lastFaces[i].y - 10
            );
            
            // 繪製文字背景
            rectangle(
                resizedFrame,
                textOrg + Point(0, baseline),
                textOrg + Point(textSize.width, -textSize.height),
                Scalar(0, 255, 0),
                FILLED
            );
            
            // 繪製文字
            putText(
                resizedFrame,
                lastLabels[i],
                textOrg,
                FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0, 0, 0),
                2
            );
        }
        
        // 顯示影像
        imshow("即時人臉辨識系統 - Real-time Face Recognition", resizedFrame);
        
        // === FR-1.6: 系統退出 ===
        char key = (char)waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' 或 ESC
            cout << "\n系統正常退出..." << endl;
            break;
        }
        
        frameCount++;
    }
    
    // 釋放資源
    camera.release();
    destroyAllWindows();
    
    cout << "資源已釋放，程式結束。" << endl;
    return 0;
}