//=============================================================================
//
// akaze_match.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_match.cpp
 * @brief Main program for matching two images with AKAZE features
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla
 */

#include "akaze.hpp"


namespace terraclear
{
  akaze::akaze() 
  { 
    
  }
  
  std::vector<cv::Point2f> akaze::find_points(cv::Mat& img1, cv::Mat& img2)
  {
    AKAZEOptions options = AKAZEOptions();
    // Convert the images to float
    img1.convertTo(img1_32, CV_32F, 1.0/255.0, 0);
    img2.convertTo(img2_32, CV_32F, 1.0/255.0, 0);
    // Color images for results visualization
    img1_rgb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
    img2_rgb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
    img_com = cv::Mat(cv::Size(img1.cols*2, img1.rows), CV_8UC3);
    img_r = cv::Mat(cv::Size(img_com.cols*rfactor, img_com.rows*rfactor), CV_8UC3);
    // Create the first AKAZE object
    options.img_width = img1.cols;
    options.img_height = img1.rows;
    libAKAZECU::AKAZE evolution1(options);

    // Create the second HKAZE object
    options.img_width = img2.cols;
    options.img_height = img2.rows;
    libAKAZECU::AKAZE evolution2(options);

    t1 = cv::getTickCount();

    cudaProfilerStart();
    
    evolution1.Create_Nonlinear_Scale_Space(img1_32);
    evolution1.Feature_Detection(kpts1);
    evolution1.Compute_Descriptors(kpts1, desc1);

    evolution2.Create_Nonlinear_Scale_Space(img2_32);
    evolution2.Feature_Detection(kpts2);
    evolution2.Compute_Descriptors(kpts2, desc2);

    t2 = cv::getTickCount();
    takaze = 1000.0*(t2-t1)/cv::getTickFrequency();

    // Matching Descriptors!!
    std::vector<cv::Point2f> matches, inliers;

    t1 = cv::getTickCount();
    // Create OpenCV cuda matcher
    auto m1 = std::make_unique<cv::cuda::DescriptorMatcher>(cv::NORM_L2);
    m1->knnMatch(desc1, desc2, dmatches, 2);
    //t2 = cv::getTickCount();
    tmatch = 1000.0*(t2 - t1)/ cv::getTickFrequency();
    
    cudaProfilerStop();

    // Compute Inliers!!
    matches2points_nndr(kpts2, kpts1, dmatches, matches, DRATIO);
    compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);

    // Compute the inliers statistics
    nkpts1 = kpts1.size();
    nkpts2 = kpts2.size();
    nmatches = matches.size()/2;
    ninliers = inliers.size()/2;
    noutliers = nmatches - ninliers;
    ratio = 100.0*((float) ninliers / (float) nmatches);

    return inliers;
  }
};