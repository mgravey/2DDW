#include <iostream>
#include <cmath>
#include "2ddw.h"
#include <chrono>



#include "opencv2/opencv.hpp"

int main(int argc, char const *argv[])
{
	fprintf(stderr, "%s\n", "start");

    //cv::Mat image1=cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    //cv::Mat image2=cv::imread(argv[2]), CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat image1= cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat image2= cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    image1.convertTo(image1, CV_32FC1, 1.0/255.0);
    image2.convertTo(image2, CV_32FC1, 1.0/255.0);

    cv::resize(image1, image1, cv::Size(image1.cols/4,image1.rows/4));
    cv::resize(image2, image2, cv::Size(image2.cols/4,image2.rows/4));

    cv::Mat image1T;
    cv::Mat image2T;
    cv::transpose(image1,image1T);
    cv::transpose(image2,image2T);

    //image2=cv::Mat(image2, cv::Rect(20,20,10,10));


    //cv::imshow("image1",image1);
    //cv::waitKey(2000);

    fprintf(stderr, "%s %d, %d\n", "size =", image1.cols, image1.rows);
    
    //start resolution
    auto begin = std::chrono::high_resolution_clock::now();

    //float distance = BiDDW((float*)image1.data,(float*)image2.data,image1.cols,image1.rows,image2.cols,image2.rows);
    float distance2 = BiDDW((float*)image1.data,(float*)image2.data,(float*)image1T.data,(float*)image2T.data,image1.cols,image1.rows,image2.cols,image2.rows);
	
    auto end = std::chrono::high_resolution_clock::now();
    double time = 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    fprintf(stderr,"compuattion time: %7.2f\n", time);
    

    fprintf(stderr, "%s %f\n", "valeur :", distance2);

	return 0;
}