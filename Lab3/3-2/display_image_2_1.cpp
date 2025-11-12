#include <fcntl.h> 
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>

using namespace cv;
using namespace std;

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // framebuffer depth
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, const char *argv[])
{    
    string inputPath = (argc >= 2) ? argv[1] : "lab3_final_24.jpg";
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    // read image file (sample.bmp) from opencv libs.
    Mat image = imread(inputPath);

    // get image size of the image.
    Size2f image_size = image.size();

    // transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
    cvtColor(image, image, COLOR_BGR2BGR565);

    // output to framebufer row by row
    for (int y = 0; y < image_size.height; y++)
    {
        // move to the next written position of output device framebuffer by "std::ostream::seekp()".
        // posisiotn can be calcluated by "y", "fb_info.xres_virtual", and "fb_info.bits_per_pixel".
        int pos = y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);  // per byte
        ofs.seekp(pos);

        // write to the framebuffer by "std::ostream::write()".
        // you could use "cv::Mat::ptr()" to get the pointer of the corresponding row.
        // you also have to count how many bytes to write to the buffer
        uchar* row_ptr = image.Mat::ptr<uchar>(y);
        int bytes_per_row = image_size.width * (fb_info.bits_per_pixel / 8);
        ofs.write(reinterpret_cast<const char*>(row_ptr), static_cast<std::streamsize>(bytes_per_row));
    }

    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open device with linux system call "open()"
    int fd = open(framebuffer_device_path, O_RDWR);
    if(fd == -1) {
        std::cerr << "Error: cannot open framebuffer device " << framebuffer_device_path << std::endl;
        fb_info.bits_per_pixel = 0;
        fb_info.xres_virtual = 0;
        return fb_info;
    }

    // get attributes of the framebuffer device thorugh linux system call "ioctl()".
    // the command you would need is "FBIOGET_VSCREENINFO"
    if(ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        std::cerr << "Error: cannot get variable screen info" << std::endl;
        close(fd);
        fb_info.bits_per_pixel = 0;
        fb_info.xres_virtual = 0;
        return fb_info;
    }

    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    // fb_info.xres_virtual =       // 8
    // fb_info.bits_per_pixel =     // 16
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    close(fd);

    return fb_info;
};