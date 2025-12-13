/**
 * Modified tool_screenshot_v2.cpp
 * Based on 5_1_main_v2.cpp architecture
 * Features:
 * 1. Mmap based Framebuffer writing (Fast)
 * 2. Separate thread for saving images (Non-blocking UI)
 * 3. Terminal raw mode for input
 */

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>
#include <termios.h>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ================= 全域變數與鎖 =================
std::mutex data_mutex;           // 保護共享資料
Mat global_frame_to_save;        // 要儲存的圖片緩衝區
bool save_request = false;       // 觸發存檔的訊號
std::atomic<bool> is_running(true); // 程式運行狀態

// 終端機設定 (用於讀取鍵盤)
static struct termios orig_termios;

// ================= 輔助函式宣告 =================
void set_terminal_mode(bool enable_raw);
int kbhit();

// ================= 存檔工作執行緒 =================
// 這個執行緒專門負責將圖片寫入 SD 卡/硬碟，避免卡住畫面
void saver_thread_func() {
    // 1. 準備存檔路徑 (與原版邏輯相同，自動建立新資料夾)
    string screenshot_dir_path;
    int dir_index = 0;
    while (true) {
        stringstream path_ss;
        // 注意：請根據您的板子實際路徑修改，這裡保留原版路徑
        path_ss << "/run/media/mmcblk1p1/screenshot_" << dir_index;
        string path_to_check = path_ss.str();
        
        struct stat st;
        if (stat(path_to_check.c_str(), &st) == -1) {
            if (mkdir(path_to_check.c_str(), 0755) == 0) {
                screenshot_dir_path = path_to_check;
                cout << "[Saver Thread] Directory created: " << screenshot_dir_path << endl;
                break;
            } else {
                screenshot_dir_path = "."; // 失敗就存當前目錄
                break;
            }
        }
        dir_index++;
    }

    int screenshot_count = 0;
    Mat local_img_to_save;

    while (is_running) {
        bool has_job = false;

        // 2. 檢查是否有存檔請求
        {
            lock_guard<mutex> lock(data_mutex);
            if (save_request) {
                // 複製圖片出來，盡快釋放鎖，讓主程式繼續跑
                local_img_to_save = global_frame_to_save.clone();
                save_request = false;
                has_job = true;
            }
        }

        // 3. 執行存檔 (耗時操作)
        if (has_job && !local_img_to_save.empty()) {
            stringstream filename_ss;
            filename_ss << screenshot_dir_path << "/" << screenshot_count << ".bmp";
            string filename = filename_ss.str();

            if (imwrite(filename, local_img_to_save)) {
                screenshot_count++;
                cout << "\r[Saved] " << filename << " (Total: " << screenshot_count << ")    " << flush;
            } else {
                cerr << "\r[Error] Failed to save " << filename << flush;
            }
        } else {
            // 沒工作就休息，避免佔用 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// ================= 主程式 =================
int main() {
    // 1. 初始化 Framebuffer (使用 mmap，參考 5_1_main_v2.cpp)
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) {
        cerr << "Error: Cannot open /dev/fb0" << endl;
        return -1;
    }

    struct fb_var_screeninfo vinfo;
    if (ioctl(fb, FBIOGET_VSCREENINFO, &vinfo)) {
        cerr << "Error: Cannot get screen info" << endl;
        close(fb);
        return -1;
    }
    
    int screen_w = vinfo.xres;
    int screen_h = vinfo.yres;
    int bpp = vinfo.bits_per_pixel;
    int screensize = screen_w * screen_h * bpp / 8;

    cout << "Screen Info: " << screen_w << "x" << screen_h << " " << bpp << "bpp" << endl;

    unsigned char* fbp = (unsigned char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
    if ((intptr_t)fbp == -1) {
        cerr << "Error: mmap failed" << endl;
        close(fb);
        return -1;
    }

    // 2. 初始化相機
    VideoCapture cap(2);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera 2" << endl;
        munmap(fbp, screensize);
        close(fb);
        return -1;
    }
    // 設定相機參數 (可選)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 3. 啟動存檔執行緒
    thread saver(saver_thread_func);

    // 設定終端機為 Raw mode 以讀取按鍵
    set_terminal_mode(true);

    cout << "Started. Press 'c' to capture, 'ESC' to exit." << endl;

    Mat frame, display_frame;
    
    while (is_running) {
        // A. 讀取畫面
        cap >> frame;
        if (frame.empty()) {
            cerr << "Empty frame!" << endl;
            break;
        }

        // B. 處理顯示 (縮放並轉為 BGR565)
        // 縮放至全螢幕
        resize(frame, display_frame, Size(screen_w, screen_h));
        // 轉色
        cvtColor(display_frame, display_frame, COLOR_BGR2BGR565);

        // C. 寫入 Framebuffer (直接記憶體拷貝，極快)
        // 假設是 16-bit (2 bytes) 色彩深度
        if (bpp == 16) {
            for (int y = 0; y < screen_h; y++) {
                // 計算記憶體偏移量
                long int location = (y * screen_w) * 2;
                // 使用 memcpy 複製一整行
                memcpy(fbp + location, display_frame.ptr(y), screen_w * 2);
            }
        } else {
            // 如果不是 16bit，這裡需要額外處理 (大多數開發板是 16bit)
            // 這裡簡單防呆
        }

        // D. 檢查按鍵輸入
        if (kbhit()) {
            int key = getchar();
            
            if (key == 27) { // ESC
                is_running = false;
            } else if (key == 'c' || key == 'C') {
                // 觸發存檔請求
                lock_guard<mutex> lock(data_mutex);
                if (!save_request) { // 如果上一次存檔還沒處理完，就忽略這次按鍵(防連點)
                    global_frame_to_save = frame.clone(); // 存原始高畫質圖
                    save_request = true;
                    // cout << "Request sent." << endl; 
                }
            }
        }
        
        // 控制 FPS，避免吃滿 CPU
        // usleep(1000); 
    }

    // 清理資源
    set_terminal_mode(false);
    cout << "\nWaiting for saver thread to finish..." << endl;
    saver.join(); // 等待存檔執行緒結束
    
    munmap(fbp, screensize);
    close(fb);
    cout << "Program exited." << endl;

    return 0;
}

// ================= 終端機控制函式 =================
void set_terminal_mode(bool enable_raw) {
    if (enable_raw) {
        tcgetattr(STDIN_FILENO, &orig_termios);
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ICANON | ECHO); 
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    } else {
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
    }
}

int kbhit() {
    struct timeval tv = {0, 0};
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    return select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv) > 0;
}