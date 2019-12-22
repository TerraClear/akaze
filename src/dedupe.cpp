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

#include "dedupe.hpp"


namespace terraclear
{
  dedupe::dedupe() 
  { 
    
  }
  
  std::vector<cv::Point2f> dedupe::find_points(cv::Mat& img1, cv::Mat& img2)
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
    cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");
    cv::Ptr<cv::DescriptorMatcher> matcher_l1 = cv::DescriptorMatcher::create("BruteForce-Hamming");

    t1 = cv::getTickCount();

    libAKAZECU::Matcher cuda_matcher;

    cuda_matcher.bfmatch(desc1, desc2, dmatches);
    cuda_matcher.bfmatch(desc2, desc1, dmatches);
    //MatchDescriptors(desc1, desc2, dmatches);

    //std::cout << "#matches: " << dmatches.size() << std::endl;
    //std::cout << "#kptsq:   " << kpts1.size() << std::endl;
    //std::cout << "#kptst:   " << kpts2.size() << std::endl;
    
    cudaProfilerStop();
    
    t2 = cv::getTickCount();
    tmatch = 1000.0*(t2 - t1)/ cv::getTickFrequency();

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

    // Show matching statistics
    /*
    std::cout << "Number of Keypoints Image 1: " << nkpts1 << std::endl;;
    std::cout << "Number of Keypoints Image 2: " << nkpts2 << std::endl;;
    std::cout << "A-KAZE Features Extraction Time (ms): " << takaze << std::endl;;
    std::cout << "Matching Descriptors Time (ms): " << tmatch << std::endl;;
    std::cout << "Number of Matches: " << nmatches << std::endl;;
    std::cout << "Number of Inliers: " << ninliers << std::endl;;
    std::cout << "Number of Outliers: " << noutliers << std::endl;;
    std::cout << "Inliers Ratio: " << ratio << std::endl; 
    */

    //draw_keypoints(img1_rgb, kpts1);
    //draw_keypoints(img2_rgb, kpts2);
    //draw_inliers(img1_rgb, img2_rgb, img_com, inliers);
    //cv::namedWindow("Inliers", cv::WINDOW_NORMAL);
    //cv::imshow("Inliers",img_com);
    //cv::Mat save_img;
    //cv::resize(img_com, save_img, cv::Size(), 0.5,0.5);
    //cv::imwrite("result.jpg", img_com);
    //cv::waitKey(0);

    return inliers;
  }
};