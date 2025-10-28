#include <opencv2/opencv.hpp>
#include <iostream>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <cstring>

int main() {
    // 打開 framebuffer
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) {
        std::cerr << "無法開啟 framebuffer (/dev/fb0)" << std::endl;
        return -1;
    }

    // 取得螢幕資訊
    struct fb_var_screeninfo vinfo;
    if (ioctl(fb, FBIOGET_VSCREENINFO, &vinfo)) {
        std::cerr << "無法取得螢幕資訊" << std::endl;
        close(fb);
        return -1;
    }

    int width = vinfo.xres;
    int height = vinfo.yres;
    int bpp = vinfo.bits_per_pixel;
    int screensize = width * height * bpp / 8;

    // 記憶體映射
    unsigned char* fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
    if ((intptr_t)fbp == -1) {
        std::cerr << "mmap framebuffer 失敗" << std::endl;
        close(fb);
        return -1;
    }

    // 開啟攝影機
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "無法開啟攝影機" << std::endl;
        munmap(fbp, screensize);
        close(fb);
        return -1;
    }

    // 載入人臉偵測模型
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade\n";
        return -1;
    }

    cv::Mat frame, gray, small_frame, display;
    std::vector<cv::Rect> faces;

    // 設定偵測影像縮小尺寸
    const int DETECT_WIDTH = 320;
    const int DETECT_HEIGHT = 240;

    // 顯示小畫面位置
    const int DISP_WIDTH = 640;
    const int DISP_HEIGHT = 480;
    const int DISP_X = 0;
    const int DISP_Y = 0;

    std::cout << "Face detection started. Press ESC to exit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // 縮小影像作偵測
        cv::resize(frame, small_frame, cv::Size(DETECT_WIDTH, DETECT_HEIGHT));
        cv::cvtColor(small_frame, gray, cv::COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        // 在原始小影像畫框
        for (size_t i = 0; i < faces.size(); i++) {
            // 將偵測座標放大回原始大小
            cv::Rect r(
                faces[i].x * frame.cols / DETECT_WIDTH,
                faces[i].y * frame.rows / DETECT_HEIGHT,
                faces[i].width * frame.cols / DETECT_WIDTH,
                faces[i].height * frame.rows / DETECT_HEIGHT
            );
            cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
        }

        // 調整小畫面顯示在 framebuffer
        cv::resize(frame, display, cv::Size(DISP_WIDTH, DISP_HEIGHT));
        cv::cvtColor(display, display, cv::COLOR_BGR2BGR565);

        // 將小畫面寫入 framebuffer 左上角
        for (int y = 0; y < DISP_HEIGHT; y++) {
            memcpy(fbp + ((DISP_Y + y) * width + DISP_X) * 2, display.ptr(y), DISP_WIDTH * 2);
        }

        // ESC 離開
        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    // 清除畫面
    memset(fbp, 0, screensize);

    // 收尾
    munmap(fbp, screensize);
    close(fb);
    cap.release();

    std::cout << "已退出並清除畫面\n";
    return 0;
}
