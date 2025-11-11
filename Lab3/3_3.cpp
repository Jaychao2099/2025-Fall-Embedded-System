#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <cstring>
#include <map>
#include <string>

// 讀取 label_dict.txt
std::map<int, std::string> loadLabels(const std::string &filename) {
    std::map<int, std::string> labels;
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "❌ 無法打開標籤檔: " << filename << std::endl;
        return labels;
    }

    int id;
    std::string student_id;
    while (file >> id >> student_id) {
        labels[id] = student_id;
    }

    file.close();
    return labels;
}

int main() {
    std::map<int, std::string> label_map = loadLabels("label_dict.txt");

    // 測試
    std::map<int, std::string>::iterator it;
    for (it = label_map.begin(); it != label_map.end(); ++it) {
        std::cout << "Label " << it->first << " => " << it->second << std::endl;
    }

    // ---------------------------
    // 1️⃣ Framebuffer 初始化
    // ---------------------------
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) { std::cerr << "❌ 無法開啟 framebuffer\n"; return -1; }

    struct fb_var_screeninfo vinfo;
    if (ioctl(fb, FBIOGET_VSCREENINFO, &vinfo)) { std::cerr << "❌ 無法取得螢幕資訊\n"; close(fb); return -1; }

    int width = vinfo.xres;
    int height = vinfo.yres;
    int bpp = vinfo.bits_per_pixel;
    int screensize = width * height * bpp / 8;

    unsigned char* fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
    if ((intptr_t)fbp == -1) { std::cerr << "❌ mmap framebuffer 失敗\n"; close(fb); return -1; }

    // ---------------------------
    // 2️⃣ 攝影機初始化
    // ---------------------------
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) { std::cerr << "❌ 無法開啟攝影機\n"; munmap(fbp, screensize); close(fb); return -1; }

    // ---------------------------
    // 3️⃣ 載入人臉偵測模型
    // ---------------------------
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) { std::cerr << "❌ 無法載入 Haar 模型\n"; return -1; }

    // ---------------------------
    // 4️⃣ 載入 LBPH 辨識模型
    // ---------------------------
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
    recognizer->read("face_model.yml");

    // ---------------------------
    // 5️⃣ 主循環
    // ---------------------------
    cv::Mat frame, gray, small_frame, display;
    std::vector<cv::Rect> faces;

    const int DETECT_WIDTH = 640;
    const int DETECT_HEIGHT = 480;
    const int DISP_WIDTH = 640;
    const int DISP_HEIGHT = 480;
    const int DISP_X = 0, DISP_Y = 0;

    std::cout << "✅ Face recognition started. Press ESC to exit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 縮小影像偵測
        cv::resize(frame, small_frame, cv::Size(DETECT_WIDTH, DETECT_HEIGHT));
        cv::cvtColor(small_frame, gray, cv::COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(50, 50));

        for (size_t i = 0; i < faces.size(); i++) {
            // 放大回原圖比例
            cv::Rect r(
                faces[i].x * frame.cols / DETECT_WIDTH,
                faces[i].y * frame.rows / DETECT_HEIGHT,
                faces[i].width * frame.cols / DETECT_WIDTH,
                faces[i].height * frame.rows / DETECT_HEIGHT
            );
            cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);

            // 裁切臉部 ROI
            cv::Mat faceROI = frame(r);
            cv::Mat grayFace;
            cv::cvtColor(faceROI, grayFace, cv::COLOR_BGR2GRAY);
            cv::resize(grayFace, grayFace, cv::Size(100, 100));

            // 辨識
            int label = -1;
            double confidence = 0.0;
	    cv::equalizeHist(grayFace, grayFace); 
            recognizer->predict(grayFace, label, confidence);

            std::string name = "Unknown";
            if (confidence < 120 && label_map.count(label)) {
                name = label_map[label];
            }

            // 畫文字
            cv::putText(frame, name, cv::Point(r.x, r.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        // 輸出 framebuffer
        cv::resize(frame, display, cv::Size(DISP_WIDTH, DISP_HEIGHT));
        cv::cvtColor(display, display, cv::COLOR_BGR2BGR565);
        for (int y = 0; y < DISP_HEIGHT; y++) {
            memcpy(fbp + ((DISP_Y + y) * width + DISP_X) * 2, display.ptr(y), DISP_WIDTH * 2);
        }

        char c = cv::waitKey(1);
        if (c == 27) break;  // ESC 離開
    }

    // 清除 framebuffer
    memset(fbp, 0, screensize);
    munmap(fbp, screensize);
    close(fb);
    cap.release();

    std::cout << " 已退出並清除畫面\n";
    return 0;
}
