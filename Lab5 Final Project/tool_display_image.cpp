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
    uint32_t xres;              // visible resolution x
    uint32_t yres;              // visible resolution y
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, const char *argv[])
{    
    string inputPath = (argc >= 2) ? argv[1] : "lab3_final_24.jpg";
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    
    if(fb_info.xres == 0 || fb_info.yres == 0) {
        cerr << "Error: Invalid framebuffer info" << endl;
        return -1;
    }
    
    cout << "Framebuffer resolution: " << fb_info.xres << "x" << fb_info.yres << endl;
    cout << "Bits per pixel: " << fb_info.bits_per_pixel << endl;
    
    std::ofstream ofs("/dev/fb0");
    if(!ofs.is_open()) {
        cerr << "Error: Cannot open /dev/fb0 for writing" << endl;
        return -1;
    }

    // read image file
    Mat image = imread(inputPath);
    if(image.empty()) {
        cerr << "Error: Cannot read image file " << inputPath << endl;
        return -1;
    }

    // get original image size
    Size2f image_size = image.size();
    cout << "Original image size: " << image_size.width << "x" << image_size.height << endl;

    // resize image to fit framebuffer resolution
    Mat resized_image;
    Size target_size(fb_info.xres, fb_info.yres);
    
    // Option 1: Resize to exact screen size (may distort aspect ratio)
    // resize(image, resized_image, target_size, 0, 0, INTER_LINEAR);
    
    // Option 2: Resize keeping aspect ratio (recommended)
    double scale_x = (double)fb_info.xres / image_size.width;
    double scale_y = (double)fb_info.yres / image_size.height;
    double scale = min(scale_x, scale_y);  // keep aspect ratio, fit in screen
    
    Size scaled_size((int)(image_size.width * scale), (int)(image_size.height * scale));
    resize(image, resized_image, scaled_size, 0, 0, INTER_LINEAR);
    
    cout << "Resized image size: " << scaled_size.width << "x" << scaled_size.height << endl;
    
    // Create a black canvas of framebuffer size and center the resized image
    Mat display_image = Mat::zeros(fb_info.yres, fb_info.xres, image.type());
    int offset_x = (fb_info.xres - scaled_size.width) / 2;
    int offset_y = (fb_info.yres - scaled_size.height) / 2;
    
    // Copy resized image to center of display canvas
    Rect roi(offset_x, offset_y, scaled_size.width, scaled_size.height);
    resized_image.copyTo(display_image(roi));

    // transfer color space from BGR to BGR565 (16-bit image)
    cvtColor(display_image, display_image, COLOR_BGR2BGR565);

    // output to framebuffer row by row
    for (int y = 0; y < fb_info.yres; y++)
    {
        int pos = y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
        ofs.seekp(pos);

        uchar* row_ptr = display_image.ptr<uchar>(y);
        int bytes_per_row = fb_info.xres * (fb_info.bits_per_pixel / 8);
        ofs.write(reinterpret_cast<const char*>(row_ptr), static_cast<std::streamsize>(bytes_per_row));
    }

    // cout << "Image displayed successfully!" << endl;
    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;

    int fd = open(framebuffer_device_path, O_RDWR);
    if(fd == -1) {
        cerr << "Error: cannot open framebuffer device " << framebuffer_device_path << endl;
        fb_info.bits_per_pixel = 0;
        fb_info.xres_virtual = 0;
        fb_info.xres = 0;
        fb_info.yres = 0;
        return fb_info;
    }

    if(ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1) {
        cerr << "Error: cannot get variable screen info" << endl;
        close(fd);
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