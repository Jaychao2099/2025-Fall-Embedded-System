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

using namespace cv;
using namespace cv::face;
using namespace std;

// 全域變數用於信號處理
volatile sig_atomic_t running = 1;

// 信號處理函數
void signalHandler(int signum) {
    cout << "\nready to exit..." << endl;
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
            throw runtime_error("cannot read fixed screen info");
        }
        
        // 獲取可變螢幕資訊
        if (ioctl(fbFd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
            close(fbFd);
            throw runtime_error("cannot read variable screen info");
        }
        
        // 計算螢幕大小
        screensize = vinfo.yres_virtual * finfo.line_length;
        
        // 映射 framebuffer 到記憶體
        fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbFd, 0);
        if (fbp == MAP_FAILED) {
            close(fbFd);
            throw runtime_error("cannot map framebuffer to memory");
        }
        
        cout << "  Init Framebuffer :" << endl;
        cout << "  Resolution: " << vinfo.xres << "x" << vinfo.yres << endl;
        cout << "  Color Depth: " << vinfo.bits_per_pixel << " bits" << endl;
        cout << "  Line Length: " << finfo.line_length << " bytes" << endl;
    }
    
    ~FrameBuffer() {
        if (fbp != nullptr && fbp != MAP_FAILED) {
            munmap(fbp, screensize);
        }
        if (fbFd != -1) {
            close(fbFd);
        }
    }
    
    // 顯示影像到 framebuffer
    void displayImage(const Mat& frame) {
        // 確保影像是 BGR 格式
        Mat bgrFrame;
        if (frame.channels() == 1) {
            cvtColor(frame, bgrFrame, COLOR_GRAY2BGR);
        } else {
            bgrFrame = frame;
        }
        
        // 調整影像大小以符合螢幕
        Mat resizedFrame;
        resize(bgrFrame, resizedFrame, Size(vinfo.xres, vinfo.yres));
        
        // 根據色彩深度轉換並寫入
        if (vinfo.bits_per_pixel == 32) {
            // 32 位元 (RGBA/BGRA)
            writeRGB32(resizedFrame);
        } else if (vinfo.bits_per_pixel == 24) {
            // 24 位元 (RGB/BGR)
            writeRGB24(resizedFrame);
        } else if (vinfo.bits_per_pixel == 16) {
            // 16 位元 (RGB565)
            writeRGB16(resizedFrame);
        } else {
            cerr << "unsupported color depth: " << vinfo.bits_per_pixel << endl;
        }
    }
    
    int getWidth() const { return vinfo.xres; }
    int getHeight() const { return vinfo.yres; }
    
private:
    void writeRGB32(const Mat& frame) {
        for (int y = 0; y < frame.rows; y++) {
            long location = y * finfo.line_length;
            for (int x = 0; x < frame.cols; x++) {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                *(fbp + location + x * 4 + 0) = pixel[0]; // B
                *(fbp + location + x * 4 + 1) = pixel[1]; // G
                *(fbp + location + x * 4 + 2) = pixel[2]; // R
                *(fbp + location + x * 4 + 3) = 0;        // A
            }
        }
    }
    
    void writeRGB24(const Mat& frame) {
        for (int y = 0; y < frame.rows; y++) {
            long location = y * finfo.line_length;
            for (int x = 0; x < frame.cols; x++) {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                *(fbp + location + x * 3 + 0) = pixel[2]; // R
                *(fbp + location + x * 3 + 1) = pixel[1]; // G
                *(fbp + location + x * 3 + 2) = pixel[0]; // B
            }
        }
    }
    
    void writeRGB16(const Mat& frame) {
        for (int y = 0; y < frame.rows; y++) {
            long location = y * finfo.line_length;
            for (int x = 0; x < frame.cols; x++) {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                // RGB565 格式: RRRRR GGGGGG BBBBB
                unsigned short rgb565 = 
                    ((pixel[2] >> 3) << 11) |  // R: 5 bits
                    ((pixel[1] >> 2) << 5) |   // G: 6 bits
                    (pixel[0] >> 3);           // B: 5 bits
                *((unsigned short*)(fbp + location + x * 2)) = rgb565;
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
        // === FR-1.1: 系統初始化 ===
        
        // 1. 載入 Haar Cascade 人臉偵測器
        string cascadePath = "haarcascade_frontalface_default.xml";
        CascadeClassifier faceCascade;
        
        if (!faceCascade.load(cascadePath)) {
            cerr << "Error: cannot load model " << cascadePath << endl;
            return -1;
        }
        cout << "load model: " << cascadePath << " success" << endl;

        // 2. 載入 LBPH 人臉辨識模型
        string modelPath = "lbph_model.yml";
        Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
        
        try {
            recognizer->read(modelPath);
            cout << "load model: " << modelPath << " success" << endl;
        } catch (Exception& e) {
            cerr << "Error: cannot load model " << modelPath << endl;
            cerr << "Details: " << e.what() << endl;
            return -1;
        }
        
        // 3. 初始化 Framebuffer
        FrameBuffer fb("/dev/fb0");
        
        // 4. 初始化攝影機
        VideoCapture camera(2);
        
        if (!camera.isOpened()) {
            cerr << "Error: cannot open camera" << endl;
            return -1;
        }
        cout << "open camera: success" << endl;

        // 建立 ID 對應表
        map<int, string> idToStudentId = createIdToStudentIdMap();
        
        // 效能優化參數
        const Size TARGET_SIZE(fb.getWidth(), fb.getHeight());
        const int FRAME_SKIP = 2;
        const double CONFIDENCE_THRESHOLD = 100.0;
        
        int frameCount = 0;
        vector<Rect> lastFaces;
        vector<string> lastLabels;

        cout << "\nsystem started, press Ctrl+C to exit...\n" << endl;

        // === 主迴圈 ===
        while (running) {
            // === FR-1.2: 影像擷取與預處理 ===
            Mat frame, resizedFrame, grayFrame;
            
            camera >> frame;
            if (frame.empty()) {
                cerr << "Warning: cannot read frame" << endl;
                break;
            }
            
            resize(frame, resizedFrame, TARGET_SIZE);
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
                    Mat faceROI = grayFrame(faces[i]);
                    
                    int predictedLabel = -1;
                    double confidence = 0.0;
                    recognizer->predict(faceROI, predictedLabel, confidence);
                    
                    string label;
                    if (confidence < CONFIDENCE_THRESHOLD && 
                        idToStudentId.find(predictedLabel) != idToStudentId.end()) {
                        label = idToStudentId[predictedLabel];
                    } else {
                        label = "unknown";
                    }
                    
                    label += " (" + to_string((int)confidence) + ")";
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
            usleep(30000); // 約 33 FPS
        }
        
        // 釋放資源
        camera.release();
        cout << "release resources, program ended." << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}