#include "./lib/AKAZE.h"

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <cuda_profiler_api.h>
#include <jsoncpp/json/json.h>

#include <iostream>
#include <fstream>

#ifndef DEDUPE_HPP
#define DEDUPE_HPP


namespace terraclear 
{
   
    class dedupe
    {
        private:
            // Variables
            cv::Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
            std::string img_path1, img_path2, homography_path;
            float ratio = 0.0, rfactor = .60;
            int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

            std::vector<cv::KeyPoint> kpts1, kpts2;
            std::vector<std::vector<cv::DMatch> > dmatches;
            cv::Mat desc1, desc2;
            cv::Mat HG;

            // Variables for measuring computation times
            double t1 = 0.0, t2 = 0.0;
            double takaze = 0.0, tmatch = 0.0;

            // Image matching options
            const float MIN_H_ERROR = 100.50f;            // Maximum error in pixels to accept an inlier
            const float DRATIO = 0.8f;                 // NNDR Matching value
        
        public:
            // Function to compute homography based off inlier points
            dedupe();
            std::vector<cv::Point2f> find_points(cv::Mat&, cv::Mat&);

    };
}

#endif /* DEDUPE_HPP */
