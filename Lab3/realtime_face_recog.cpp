#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <map>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <signal.h>
#include <getopt.h> // 新增：用於解析命令列參數

using namespace cv;
using namespace cv::face;
using namespace std;

// 全域變數用於信號處理
volatile sig_atomic_t running = 1;

// 信號處理函數
void signalHandler(int signum) {
    cout << "\nexiting..." << endl;
    running = 0;
}

// Frame Buffer 類別
class FrameBuffer {
private:
    int fbFd;
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    unsigned char* fbp;
    long screensize;
    
public:
    FrameBuffer(const char* device = "/dev/fb0") : fbFd(-1), fbp(nullptr) {
        // 開啟 framebuffer 設備
        fbFd = open(device, O_RDWR);
        if (fbFd == -1) {
            throw runtime_error("cannot open framebuffer device: " + string(device));
        }
        
        // 獲取固定螢幕資訊
        if (ioctl(fbFd, FBIOGET_FSCREENINFO, &finfo) == -1) {
            close(fbFd);
            throw runtime_error("cannot read fixed screen information");
        }
        
        // 獲取可變螢幕資訊
        if (ioctl(fbFd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
            close(fbFd);
            throw runtime_error("cannot read variable screen information");
        }
        
        // 計算螢幕大小
        screensize = vinfo.yres_virtual * finfo.line_length;
        
        // 映射 framebuffer 到記憶體
        fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbFd, 0);
        if (fbp == MAP_FAILED) {
            close(fbFd);
            throw runtime_error("cannot map framebuffer to memory");
        }
        
        // cout << "Framebuffer 初始化成功:" << endl;
        // cout << "  解析度: " << vinfo.xres << "x" << vinfo.yres << endl;
        // cout << "  色彩深度: " << vinfo.bits_per_pixel << " bits" << endl;
        // cout << "  行長度: " << finfo.line_length << " bytes" << endl;
    }
    
    ~FrameBuffer() {
        if (fbp != nullptr && fbp != MAP_FAILED) {
            munmap(fbp, screensize);
        }
        if (fbFd != -1) {
            close(fbFd);
        }
    }
    
    // 顯示影像到 framebuffer（置中顯示）
    void displayImage(const Mat& frame) {
        // 確保影像是 BGR 格式
        Mat bgrFrame;
        if (frame.channels() == 1) {
            cvtColor(frame, bgrFrame, COLOR_GRAY2BGR);
        } else {
            bgrFrame = frame;
        }
        
        // 計算置中位置
        int offsetX = (vinfo.xres - bgrFrame.cols) / 2;
        int offsetY = (vinfo.yres - bgrFrame.rows) / 2;
        
        // 確保偏移量不為負
        offsetX = max(0, offsetX);
        offsetY = max(0, offsetY);
        
        // 清空螢幕（填充黑色）
        clearScreen();
        
        // 根據色彩深度轉換並寫入（置中）
        if (vinfo.bits_per_pixel == 32) {
            writeRGB32Centered(bgrFrame, offsetX, offsetY);
        } else if (vinfo.bits_per_pixel == 24) {
            writeRGB24Centered(bgrFrame, offsetX, offsetY);
        } else if (vinfo.bits_per_pixel == 16) {
            writeRGB16Centered(bgrFrame, offsetX, offsetY);
        } else {
            cerr << "unsupported color depth: " << vinfo.bits_per_pixel << endl;
        }
    }
    
    int getWidth() const { return vinfo.xres; }
    int getHeight() const { return vinfo.yres; }
    
private:
    void clearScreen() {
        memset(fbp, 0, screensize);
    }
    
    void writeRGB32Centered(const Mat& frame, int offsetX, int offsetY) {
        for (int y = 0; y < frame.rows && (y + offsetY) < (int)vinfo.yres; y++) {
            long location = (y + offsetY) * finfo.line_length;
            for (int x = 0; x < frame.cols && (x + offsetX) < (int)vinfo.xres; x++) {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                long pos = location + (x + offsetX) * 4;
                *(fbp + pos + 0) = pixel[0]; // B
                *(fbp + pos + 1) = pixel[1]; // G
                *(fbp + pos + 2) = pixel[2]; // R
                *(fbp + pos + 3) = 0;        // A
            }
        }
    }
    
    void writeRGB24Centered(const Mat& frame, int offsetX, int offsetY) {
        for (int y = 0; y < frame.rows && (y + offsetY) < (int)vinfo.yres; y++) {
            long location = (y + offsetY) * finfo.line_length;
            for (int x = 0; x < frame.cols && (x + offsetX) < (int)vinfo.xres; x++) {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                long pos = location + (x + offsetX) * 3;
                *(fbp + pos + 0) = pixel[2]; // R
                *(fbp + pos + 1) = pixel[1]; // G
                *(fbp + pos + 2) = pixel[0]; // B
            }
        }
    }
    
    void writeRGB16Centered(const Mat& frame, int offsetX, int offsetY) {
        for (int y = 0; y < frame.rows && (y + offsetY) < (int)vinfo.yres; y++) {
            long location = (y + offsetY) * finfo.line_length;
            for (int x = 0; x < frame.cols && (x + offsetX) < (int)vinfo.xres; x++) {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                // RGB565 格式: RRRRR GGGGGG BBBBB
                unsigned short rgb565 = 
                    ((pixel[2] >> 3) << 11) |  // R: 5 bits
                    ((pixel[1] >> 2) << 5) |   // G: 6 bits
                    (pixel[0] >> 3);           // B: 5 bits
                long pos = location + (x + offsetX) * 2;
                *((unsigned short*)(fbp + pos)) = rgb565;
            }
        }
    }
};

// 定義 ID 到學號的對應表
map<int, string> createIdToStudentIdMap() {
    map<int, string> idMap;
    idMap[0] = "314552021";
    idMap[1] = "314554053";
    return idMap;
}

int main(int argc, char** argv) {
    // 設定信號處理器 (Ctrl+C)
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        // --- 新增：命令列參數解析（-m <model_path> -t <threshold> -h） ---
        string modelPath = "lbph_model.yml";                 // 預設 LBPH 模型
        double CONFIDENCE_THRESHOLD = 120.0;                 // 預設閾值
        string cascadePath = "haarcascade_frontalface_default.xml"; // 可保留預設 cascade
        
        int opt;
        while ((opt = getopt(argc, argv, "m:t:h")) != -1) {
            switch (opt) {
                case 'm':
                    if (optarg) modelPath = string(optarg);
                    break;
                case 't':
                    if (optarg) CONFIDENCE_THRESHOLD = atof(optarg);
                    break;
                case 'h':
                default:
                    cout << "Usage: " << argv[0] << " [-m <model_path>] [-t <confidence_threshold>]\n";
                    cout << "  -m  path to LBPH model (default: lbph_model.yml)\n";
                    cout << "  -t  confidence threshold (default: 120.0)\n";
                    return 0;
            }
        }
        // 顯示所使用的設定
        cout << "Using model: " << modelPath << ", confidence threshold: " << CONFIDENCE_THRESHOLD << endl;
        // --- 參數解析結束 ---
        
        // === FR-1.1: 系統初始化 ===
        
        // 1. 載入 Haar Cascade 人臉偵測器
        CascadeClassifier faceCascade;
        
        if (!faceCascade.load(cascadePath)) {
            cerr << "error: cannot load model " << cascadePath << endl;
            return -1;
        }
        cout << "load model: " << cascadePath << " success" << endl;
        
        // 2. 載入 LBPH 人臉辨識模型
        Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
        
        try {
            recognizer->read(modelPath);
            cout << "load model: " << modelPath << " success" << endl;
        } catch (Exception& e) {
            cerr << "error: cannot load model " << modelPath << endl;
            cerr << "details: " << e.what() << endl;
            return -1;
        }
        
        // 3. 初始化 Framebuffer
        FrameBuffer fb("/dev/fb0");
        
        // 4. 初始化攝影機
        VideoCapture camera(2);
        
        if (!camera.isOpened()) {
            cerr << "error: cannot open camera" << endl;
            return -1;
        }
        cout << "open camera: success" << endl;

        // 建立 ID 對應表
        map<int, string> idToStudentId = createIdToStudentIdMap();
        
        // 效能優化參數
        const Size TARGET_SIZE(640, 480);  // 使用攝影機原始大小，不拉伸
        const int FRAME_SKIP = 3;
        
        int frameCount = 0;
        vector<Rect> lastFaces;
        vector<string> lastLabels;

        cout << "\nsystem starting, press Ctrl+C to exit...\n" << endl;

        // === 主迴圈 ===
        while (running) {
            // === FR-1.2: 影像擷取與預處理 ===
            Mat frame, resizedFrame, grayFrame;
            
            camera >> frame;
            if (frame.empty()) {
                cerr << "warning: cannot read image" << endl;
                break;
            }
            
            // 調整為固定大小但保持比例
            resize(frame, resizedFrame, TARGET_SIZE);
            // 轉換為灰階（用於偵測與辨識）
            cvtColor(resizedFrame, grayFrame, COLOR_BGR2GRAY);
            
            bool shouldProcess = (frameCount % (FRAME_SKIP + 1) == 0);
            
            if (shouldProcess) {
                // === FR-1.3: 人臉偵測 ===
                vector<Rect> faces;
                faceCascade.detectMultiScale(
                    grayFrame,
                    faces,
                    1.1,
                    5,
                    0,
                    Size(30, 30)
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
                    
                    string name;
                    if (confidence < CONFIDENCE_THRESHOLD && 
                        idToStudentId.find(predictedLabel) != idToStudentId.end()) {
                        name = idToStudentId[predictedLabel];
                    } else {
                        name = "unknown";
                    }
                    
                    // 附加信賴度資訊（可選，用於除錯）
                    // label += " (" + to_string((int)confidence) + ")";
                    // 組合最終要顯示的字串，包含名稱、原始標籤和信賴度
                    string label = name + " (L:" + to_string(predictedLabel) + ", C:" + to_string((int)confidence) + ")";
                    labels.push_back(label);
                }
                
                lastFaces = faces;
                lastLabels = labels;
            }
            
            // === FR-1.5: 結果視覺化 ===
            for (size_t i = 0; i < lastFaces.size() && i < lastLabels.size(); i++) {
                rectangle(resizedFrame, lastFaces[i], Scalar(0, 255, 0), 2);
                
                int baseline = 0;
                Size textSize = getTextSize(
                    lastLabels[i], 
                    FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    2, 
                    &baseline
                );
                
                Point textOrg(lastFaces[i].x, lastFaces[i].y - 10);
                
                rectangle(
                    resizedFrame,
                    textOrg + Point(0, baseline),
                    textOrg + Point(textSize.width, -textSize.height),
                    Scalar(0, 255, 0),
                    FILLED
                );
                
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
            
            // 顯示到 framebuffer
            fb.displayImage(resizedFrame);
            
            frameCount++;
            
            // 短暫延遲以控制幀率
            usleep(30000); // 約 30 FPS
        }
        
        // 釋放資源
        camera.release();
        cout << "release resources, program ends." << endl;
        
    } catch (const exception& e) {
        cerr << "error: " << e.what() << endl;
        return -1;
    }

    return 0;
}