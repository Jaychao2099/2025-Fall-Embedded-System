#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono> // 用於更精確的時間控制

// --- 常量與結構體定義 ---

#define FRAMEBUFFER_PATH "/dev/fb0"

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual; // 虛擬解析度寬度 (用於計算行偏移)
    uint32_t xres;         // 實際可視解析度寬度
    uint32_t yres;         // 實際可視解析度高度
};

// --- 函式宣告 ---

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);
void set_terminal_mode(bool enable_raw);
int kbhit();
void clear_framebuffer(std::fstream& ofs, const framebuffer_info& fb_info, const std::vector<char>& black_buffer);

// --- 全域變數 ---
// 保存原始終端設置
static struct termios orig_termios;

// -------------------- MAIN 函式 --------------------

int main(int argc, const char *argv[]) {
    // 1. 獲取 Framebuffer 資訊
    struct framebuffer_info fb_info = get_framebuffer_info(FRAMEBUFFER_PATH);
    if (fb_info.bits_per_pixel == 0) {
        std::cerr << "Failed to get framebuffer info or open " << FRAMEBUFFER_PATH << std::endl;
        return 1;
    }
    int bytes_per_pixel = fb_info.bits_per_pixel / 8;

    // 2. 開啟 Framebuffer 檔案
    std::fstream ofs(FRAMEBUFFER_PATH, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Could not open framebuffer device " << FRAMEBUFFER_PATH << std::endl;
        return 1;
    }
    
    // 3. 預先計算黑色緩衝區
    long total_bytes = (long)fb_info.yres * fb_info.xres_virtual * bytes_per_pixel;
    std::vector<char> black_buffer(total_bytes, 0);
    
    // 啟動時清除一次 framebuffer
    clear_framebuffer(ofs, fb_info, black_buffer);

    // 4. 載入圖片 (Picture Mode 資源)
    // cv::Mat image = cv::imread("advance.png");
    cv::Mat image = cv::imread("/run/media/mmcblk1p1/advance.png");
    bool is_test_pattern = false; // 新增旗標用於偵錯

    if (image.empty()) {
        std::cerr << "Warning: Failed to read advance.png. Initializing a blue test pattern." << std::endl;
        is_test_pattern = true;
        
        // 使用 FB 寬高作為測試圖的目標尺寸
        int test_width = fb_info.xres > 0 ? fb_info.xres : 1280; 
        int test_height = fb_info.yres > 0 ? fb_info.yres : 720;

        // 生成一個純黑色的 Mat，然後在中央繪製一個藍色方塊 (CV_8UC3: BGR 8位元)
        int square_size = std::min(test_width, test_height) / 2;
        image = cv::Mat(test_height, test_width, CV_8UC3, cv::Scalar(0, 0, 0)); // 黑色背景
        
        // 繪製一個居中的藍色矩形 (BGR order: B=255, G=0, R=0)
        cv::Rect blue_rect(
            (test_width - square_size) / 2, 
            (test_height - square_size) / 2, 
            square_size, 
            square_size
        );
        cv::rectangle(image, blue_rect, cv::Scalar(255, 0, 0), cv::FILLED); // 純藍色
        
        std::cout << "DEBUG: advance.png missing. Displaying a centered blue square (Test Pattern)." << std::endl;
    }
    
    cv::Mat image_16bit;
    if (!image.empty()) {
        // --- 圖片解析度為 3840x1080，不再進行縮放，直接用於環繞滾動 ---
        // 轉換圖片為 BGR565 (16-bit) 格式
        cv::cvtColor(image, image_16bit, cv::COLOR_BGR2BGR565);
    }

    // 5. 初始化狀態與控制
    // 5.2 預設為圖片模式 (電子滾動板)
    int scroll_x = 0;
    int image_width = image_16bit.cols;  // 圖片寬度

    // *** 5.2 自動滾動相關變數 ***
    // 預設為 1 (向右)，符合啟動時向右跑的要求
    int auto_scroll_speed = 100; // 滾動速度 (像素/幀)
    int auto_scroll_direction = 1; // 1: 向右 (L), -1: 向左 (J)

    int screenshot_count = 0;
    bool running = true;

    // 6. 設置終端為原始模式
    set_terminal_mode(true);
    std::cout << "Display App Initialized." << std::endl;
    std::cout << "Framebuffer: " << fb_info.xres << "x" << fb_info.yres << std::endl;
    std::cout << "Image: " << image_width << "x" << image_16bit.rows << std::endl;
    std::cout << "Controls: J (Scroll Left), L (Scroll Right), ESC (Exit)" << std::endl;

    // 使用 chrono 進行更精確的幀率控制
    // auto last_frame_time = std::chrono::high_resolution_clock::now();
    const int target_fps = 60; // 提高目標幀率到 60 FPS
    const std::chrono::microseconds frame_duration(1000000 / target_fps);

    // --- 主循環 ---
    while (running) {
        auto frame_start = std::chrono::high_resolution_clock::now();
            
        // --- 5.2 *** 更新滾動位置（使用模運算實現循環）***
        if (!is_test_pattern) {
            scroll_x += auto_scroll_speed * auto_scroll_direction;
            
            // 確保 scroll_x 在 [0, image_width) 範圍內
            scroll_x = ((scroll_x % image_width) + image_width) % image_width;
        }
        
        // 圖片繪圖
        if (!image_16bit.empty()) {
            int display_width = fb_info.xres;
            int display_height = fb_info.yres;
            
            // 垂直置中 (圖片高度為 1080 或測試圖的高度)
            int offset_y = (display_height - (int)image_16bit.rows) / 2;
            if (offset_y < 0) offset_y = 0;
            
            int crop_height = std::min((int)image_16bit.rows, (int)display_height);
            
            // 檢查是否需要分兩段繪製（跨越圖片邊界）
            if (scroll_x + display_width <= image_width) {
                // 情況 1: 不跨越邊界，直接繪製一段
                int roi_width = std::min(display_width, image_width - scroll_x);
                // 從圖片中裁剪出當前顯示區域
                cv::Mat cropped = image_16bit(cv::Rect(scroll_x, 0, roi_width, crop_height));

                // *** 優化：批次寫入裁剪後的圖片到 Framebuffer，減少 seekp 調用 ***
                for (int y = 0; y < cropped.rows; y++) {
                    int fb_y = y + offset_y;
                    // 使用 xres_virtual 進行行偏移計算
                    long pos = (long)fb_y * fb_info.xres_virtual * bytes_per_pixel;
                    ofs.seekp(pos);
                    
                    uchar* row_ptr = cropped.ptr<uchar>(y);
                    int bytes_to_write = cropped.cols * bytes_per_pixel;
                    ofs.write(reinterpret_cast<char*>(row_ptr), bytes_to_write);
                }
            } else {
                // 情況 2: 跨越邊界，需要分兩段繪製
                
                // 第一段：圖片末端（從 scroll_x 到圖片結束）
                int first_part_width = image_width - scroll_x;
                cv::Mat first_part = image_16bit(cv::Rect(scroll_x, 0, first_part_width, crop_height));
                
                // 第二段：圖片開頭（從 0 開始，填滿剩餘螢幕）
                int second_part_width = display_width - first_part_width;
                cv::Mat second_part = image_16bit(cv::Rect(0, 0, second_part_width, crop_height));
                
                // 繪製第一段和第二段
                for (int y = 0; y < crop_height; y++) {
                    int fb_y = y + offset_y;
                    long pos = (long)fb_y * fb_info.xres_virtual * bytes_per_pixel;
                    ofs.seekp(pos);
                    
                    // 寫入第一段
                    uchar* first_row_ptr = first_part.ptr<uchar>(y);
                    ofs.write(reinterpret_cast<char*>(first_row_ptr), first_part_width * bytes_per_pixel);
                    
                    // 寫入第二段
                    uchar* second_row_ptr = second_part.ptr<uchar>(y);
                    ofs.write(reinterpret_cast<char*>(second_row_ptr), second_part_width * bytes_per_pixel);
                }
            }
        }

        ofs.flush();

        // 7. 處理按鍵輸入
        if (kbhit()) {
            char key = getchar();

            switch (key) {
                case 27: // ESC
                    std::cout << "\nExiting..." << std::endl;
                    running = false;
                    break;
                case 'J': case 'j':
                    // 設置自動滾動方向為左 (-1)
                    auto_scroll_direction = 1;
                    break;
                case 'L': case 'l':
                    // 設置自動滾動方向為右 (1)
                    auto_scroll_direction = -1;
                    break;
            }
        }

        // *** 優化：精確的幀率控制 ***
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        
        if (elapsed < frame_duration) {
            std::chrono::microseconds sleep_time = frame_duration - elapsed;
            usleep(sleep_time.count());
        }
    }

    // 清理資源
    set_terminal_mode(false); // 恢復終端設置
    ofs.close();

    std::cout << "\nTotal screenshots taken: " << screenshot_count << std::endl;

    return 0;
}

// -------------------- 函式定義 --------------------

// 獲取 Framebuffer 資訊
struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path) {
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;

    int fd = open(framebuffer_device_path, O_RDWR);
    if (fd == -1) {
        std::cerr << "Error: cannot open framebuffer device " << framebuffer_device_path << std::endl;
        // 修正 C++11 警告：使用手動賦值
        fb_info.bits_per_pixel = 0;
        fb_info.xres_virtual = 0;
        fb_info.xres = 0;
        fb_info.yres = 0;
        return fb_info;
    }

    if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        std::cerr << "Error: cannot get variable screen info" << std::endl;
        close(fd);
        // 修正 C++11 警告：使用手動賦值
        fb_info.bits_per_pixel = 0;
        fb_info.xres_virtual = 0;
        fb_info.xres = 0;
        fb_info.yres = 0;
        return fb_info;
    }

    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    fb_info.xres = screen_info.xres;
    fb_info.yres = screen_info.yres;

    close(fd);
    return fb_info;
}

// 設置終端模式 (Raw Mode + Non-blocking)
void set_terminal_mode(bool enable_raw) {
    if (enable_raw) {
        // 保存當前終端設置
        tcgetattr(STDIN_FILENO, &orig_termios);

        struct termios raw = orig_termios;

        // 設置為原始模式
        raw.c_lflag &= ~(ICANON | ECHO); // 關閉規範模式和回顯
        raw.c_cc[VMIN] = 0;              // 非阻塞讀取
        raw.c_cc[VTIME] = 0;

        tcsetattr(STDIN_FILENO, TCSANOW, &raw);

        // 設置stdin為非阻塞
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    } else {
        // 恢復原始終端設置
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);

        // 恢復stdin為阻塞模式
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
    }
}

// 檢查是否有按鍵輸入
int kbhit() {
    struct timeval tv = {0, 0};
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    return select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv) > 0;
}

// 清除 framebuffer 的輔助函式
void clear_framebuffer(std::fstream& ofs, const framebuffer_info& fb_info, const std::vector<char>& black_buffer) {
    ofs.seekp(0);
    ofs.write(black_buffer.data(), black_buffer.size());
    ofs.flush();
}