#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <termios.h>
#include <sys/stat.h>
#include <string>
#include <vector> // For std::vector

struct framebuffer_info
{
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t xres;
    uint32_t yres;
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );
void set_terminal_mode(bool enable_raw);
int kbhit();

// 保存原始終端設置
static struct termios orig_termios;

int main ( int argc, const char *argv[] )
{
    cv::Mat frame;
    cv::Mat resized_frame; // *** 新增：用於存放縮放後影像的 Mat ***
    cv::Mat frame_bgr565;

    cv::VideoCapture camera (2, cv::CAP_V4L);

    struct framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

    std::ofstream ofs("/dev/fb0", std::ios::out | std::ios::binary);

    if( !camera.isOpened() )
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }

    if( !ofs.is_open() )
    {
        std::cerr << "Could not open framebuffer device." << std::endl;
        return 1;
    }

    // 原始攝影機解析度設定可以保留，以確保來源影像的穩定性
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    camera.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Camera started. Press 'c' to capture screenshot, ESC to exit." << std::endl;
    std::cout << "Framebuffer: " << fb_info.xres << "x" << fb_info.yres << std::endl;

    // 設置終端為原始模式，可以立即讀取按鍵
    set_terminal_mode(true);

    int screenshot_count = 0;
    bool running = true;

    // 決定並創建本次執行的截圖資料夾
    std::string screenshot_dir_path;
    int dir_index = 0;
    while (true) {
        std::stringstream path_ss;
        path_ss << "/run/media/mmcblk1p1/screenshot_" << dir_index;
        std::string path_to_check = path_ss.str();
        
        struct stat st;
        if (stat(path_to_check.c_str(), &st) == -1) {
            if (mkdir(path_to_check.c_str(), 0755) == 0) {
                screenshot_dir_path = path_to_check;
                std::cout << "Screenshot directory created: " << screenshot_dir_path << std::endl;
                break;
            } else {
                std::cerr << "Error creating directory: " << path_to_check << std::endl;
                screenshot_dir_path = "."; 
                break;
            }
        }
        dir_index++;
    }

    while ( running )
    {
        camera >> frame;
        
        if(frame.empty()) {
            std::cerr << "Could not grab frame." << std::endl;
            break;
        }

        // *** 修改：將影像縮放至與 framebuffer 相同大小 ***
        cv::resize(frame, resized_frame, cv::Size(fb_info.xres, fb_info.yres));

        // *** 移除：不再需要計算置中偏移量 ***
        // int offset_x = (fb_info.xres_virtual - frame_size.width) / 2;
        // int offset_y = (fb_info.yres - frame_size.height) / 2;

        // 將縮放後的影像轉換為 BGR565 格式
        cv::cvtColor(resized_frame, frame_bgr565, cv::COLOR_BGR2BGR565);

        // 首次清空 framebuffer (可保留)
        static bool first_frame = true;
        if(first_frame) {
            ofs.seekp(0);
            int total_bytes = fb_info.yres * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
            std::vector<char> black_buffer(total_bytes, 0);
            ofs.write(black_buffer.data(), total_bytes);
            first_frame = false;
        }

        // *** 修改：寫入影像到 framebuffer，以填滿整個螢幕 ***
        for ( int y = 0; y < fb_info.yres; y++ )
        {
            // 直接計算每行的起始位置，不再需要偏移
            int pos = y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
            ofs.seekp(pos);

            // 取得縮放後影像的該行指標
            uchar* row_ptr = frame_bgr565.ptr<uchar>(y);
            // 要寫入的位元組數為螢幕的寬度
            int bytes_per_row = fb_info.xres * (fb_info.bits_per_pixel / 8);
            ofs.write(reinterpret_cast<char*>(row_ptr), bytes_per_row);
        }
        
        ofs.flush();
        
        // 檢查是否有按鍵輸入
        if(kbhit()) {
            int key = getchar();
            if(key == -1) continue;
            
            if(key == 27) {
                std::cout << "\nExiting..." << std::endl;
                running = false;
            }
            else if(key == 'c' || key == 'C') {
                std::stringstream filename_ss;
                filename_ss << screenshot_dir_path << "/" << screenshot_count << ".bmp";
                
                // *** 注意：這裡儲存的是原始、未縮放的 'frame'，以保持最佳畫質 ***
                if(cv::imwrite(filename_ss.str(), frame)) {
                    screenshot_count++;
                    std::cout << "\rScreenshot saved: " << filename_ss.str() 
                              << " (Total this session: " << screenshot_count << ")" << std::flush;
                } else {
                    std::cout << "\rFailed to save screenshot: " << filename_ss.str() << std::flush;
                }
            }
        }
        
        // 短暫延遲以控制幀率
        usleep(33000); // 約30 FPS
    }

    // 恢復終端設置
    set_terminal_mode(false);

    camera.release();
    ofs.close();

    std::cout << "\nTotal screenshots taken: " << screenshot_count << std::endl;

    return 0;
}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;

    int fd = open(framebuffer_device_path, O_RDWR);
    if(fd == -1) {
        std::cerr << "Error: cannot open framebuffer device " << framebuffer_device_path << std::endl;
        exit(1); // Exit if framebuffer cannot be opened
    }

    if(ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        std::cerr << "Error: cannot get variable screen info" << std::endl;
        close(fd);
        exit(1); // Exit on error
    }

    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    fb_info.xres = screen_info.xres;
    fb_info.yres = screen_info.yres;

    close(fd);
    return fb_info;
}

void set_terminal_mode(bool enable_raw)
{
    if(enable_raw) {
        tcgetattr(STDIN_FILENO, &orig_termios);
        
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ICANON | ECHO); 
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }
    else {
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
    }
}

int kbhit()
{
    struct timeval tv = {0, 0};
    fd_set readfds;
    
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    
    return select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv) > 0;
}