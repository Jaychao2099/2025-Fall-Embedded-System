#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

using namespace cv;
using namespace std;

// 檢查路徑是否為目錄
bool isDirectory(const string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0)
        return false;
    return S_ISDIR(statbuf.st_mode);
}

// 檢查檔案是否為圖片格式
bool isImageFile(const string& filename) {
    string ext = filename.substr(filename.find_last_of(".") + 1);
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");
}

// 從單張圖片中偵測並裁切最大的人臉
bool detectAndCropFace(const Mat& image, const CascadeClassifier& faceCascade, Mat& faceROI) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    vector<Rect> faces;
    faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(30, 30));
    
    if (faces.empty()) {
        return false;
    }
    
    // 如果偵測到多張臉，選取面積最大的
    if (faces.size() > 1) {
        auto maxFace = max_element(faces.begin(), faces.end(), 
            [](const Rect& a, const Rect& b) {
                return a.area() < b.area();
            });
        faceROI = gray(*maxFace).clone();
        return true;
    }
    
    // 只有一張臉
    faceROI = gray(faces[0]).clone();
    return true;
}

// 掃描資料集目錄並收集訓練數據
bool scanDataset(const string& datasetPath, const CascadeClassifier& faceCascade,
                 vector<Mat>& faces, vector<int>& labels) {
    
    DIR* dir = opendir(datasetPath.c_str());
    if (!dir) {
        cerr << "錯誤：無法開啟資料集目錄: " << datasetPath << endl;
        return false;
    }
    
    int userId = 0;
    int totalImages = 0;
    int successfulImages = 0;
    
    cout << "開始掃描資料集目錄: " << datasetPath << endl;
    cout << "=====================================" << endl;
    
    struct dirent* entry;
    vector<string> userDirs;
    
    // 收集所有使用者目錄
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue; // 跳過隱藏檔案和 . ..
        
        string userPath = datasetPath + "/" + entry->d_name;
        if (isDirectory(userPath)) {
            userDirs.push_back(entry->d_name);
        }
    }
    closedir(dir);
    
    // 排序以確保一致性
    sort(userDirs.begin(), userDirs.end());
    
    // 遍歷所有使用者目錄
    for (const string& userName : userDirs) {
        string userPath = datasetPath + "/" + userName;
        
        cout << "\n處理使用者 [" << userName << "] (ID: " << userId << ")" << endl;
        
        int userImageCount = 0;
        int userSuccessCount = 0;
        
        DIR* userDir = opendir(userPath.c_str());
        if (!userDir) {
            cout << "  警告：無法開啟目錄 " << userPath << endl;
            continue;
        }
        
        // 遍歷該使用者的所有圖片
        while ((entry = readdir(userDir)) != NULL) {
            if (entry->d_name[0] == '.') continue;
            
            string filename = entry->d_name;
            if (!isImageFile(filename)) continue;
            
            string imagePath = userPath + "/" + filename;
            
            totalImages++;
            userImageCount++;
            
            // 讀取圖片
            Mat image = imread(imagePath);
            if (image.empty()) {
                cout << "  警告：無法讀取圖片 " << filename << endl;
                continue;
            }
            
            // 偵測並裁切人臉
            Mat faceROI;
            if (!detectAndCropFace(image, faceCascade, faceROI)) {
                cout << "  警告：未偵測到人臉 - " << filename << endl;
                continue;
            }
            
            // 檢查是否偵測到多張臉
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            vector<Rect> faces_check;
            faceCascade.detectMultiScale(gray, faces_check, 1.1, 4, 0, Size(30, 30));
            if (faces_check.size() > 1) {
                cout << "  警告：偵測到多張人臉，使用最大的 - " << filename << endl;
            }
            
            // 將人臉數據加入訓練集
            faces.push_back(faceROI);
            labels.push_back(userId);
            successfulImages++;
            userSuccessCount++;
        }
        
        closedir(userDir);
        
        cout << "  完成：成功處理 " << userSuccessCount << "/" << userImageCount << " 張圖片" << endl;
        userId++;
    }
    
    cout << "\n=====================================" << endl;
    cout << "資料集掃描完成！" << endl;
    cout << "總使用者數量: " << userId << endl;
    cout << "總圖片數量: " << totalImages << endl;
    cout << "成功處理圖片: " << successfulImages << endl;
    cout << "=====================================" << endl;
    
    if (userId < 2) {
        cerr << "錯誤：至少需要 2 個使用者的數據才能訓練模型" << endl;
        return false;
    }
    
    if (successfulImages == 0) {
        cerr << "錯誤：未找到任何有效的人臉數據" << endl;
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    cout << "=====================================" << endl;
    cout << "人臉辨識模型訓練器 v1.0" << endl;
    cout << "=====================================" << endl;
    
    // 參數檢查
    if (argc < 2) {
        cerr << "使用方式: " << argv[0] << " <資料集路徑> [輸出模型路徑]" << endl;
        cerr << "範例: " << argv[0] << " ./dataset lbph_model.yml" << endl;
        return -1;
    }
    
    string datasetPath = argv[1];
    string outputPath = (argc >= 3) ? argv[2] : "lbph_model.yml";
    
    cout << "\n設定資訊:" << endl;
    cout << "  資料集路徑: " << datasetPath << endl;
    cout << "  輸出模型路徑: " << outputPath << endl;
    cout << endl;
    
    // 載入 Haar 分類器
    CascadeClassifier faceCascade;
    string cascadePath = "haarcascade_frontalface_default.xml";
    
    if (!faceCascade.load(cascadePath)) {
        cerr << "錯誤：無法載入人臉偵測模型: " << cascadePath << endl;
        cerr << "請確保該檔案存在於程式執行目錄中" << endl;
        return -1;
    }
    cout << "✓ 成功載入人臉偵測模型" << endl;
    
    // 掃描資料集並收集訓練數據
    vector<Mat> faces;
    vector<int> labels;
    
    if (!scanDataset(datasetPath, faceCascade, faces, labels)) {
        cerr << "資料集掃描失敗，程式終止" << endl;
        return -1;
    }
    
    // 訓練 LBPH 模型
    cout << "\n開始訓練模型..." << endl;
    Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
    
    try {
        model->train(faces, labels);
        cout << "✓ 模型訓練完成" << endl;
    } catch (const Exception& e) {
        cerr << "錯誤：模型訓練失敗 - " << e.what() << endl;
        return -1;
    }
    
    // 儲存模型
    cout << "\n儲存模型到: " << outputPath << endl;
    try {
        model->save(outputPath);
        cout << "✓ 模型已成功儲存" << endl;
    } catch (const Exception& e) {
        cerr << "錯誤：模型儲存失敗 - " << e.what() << endl;
        return -1;
    }
    
    cout << "\n=====================================" << endl;
    cout << "訓練完成！模型已準備好供辨識系統使用。" << endl;
    cout << "=====================================" << endl;
    
    return 0;
}