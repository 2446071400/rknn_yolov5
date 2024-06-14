#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov5_model(model_path, &rknn_app_ctx);
    // if (ret != 0)
    // {
    //     printf("init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path);
    //     goto out;
    // }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    //ret = read_image(image_path, &src_image);
    cv::Mat source_img = cv::imread("../model/bus.jpg");
    cv::Mat img;
    cv::cvtColor(source_img, img, cv::COLOR_BGR2RGB);
    cv::Mat out_img(img); //放前面，放后面不让 goto
    src_image.width = img.cols;
    src_image.height = img.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = img.cols*img.rows*3;
    src_image.virt_addr = img.data;

    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;

    ret = inference_yolov5_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("init_yolov5_model fail! ret=%d\n", ret);
        goto out;
    }

    // 画框和概率
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_YELLOW, 10);
    }

    //write_image("out.png", &src_image);
    out_img.data = src_image.virt_addr;
    cv::cvtColor(out_img, out_img, cv::COLOR_BGR2RGB);
    cv::imwrite("out_opencv.png", out_img);

out:
    deinit_post_process();

    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov5_model fail! ret=%d\n", ret);
    }

    // if (src_image.virt_addr != NULL)
    // {
    //     free(src_image.virt_addr);
    // }

    return 0;
}