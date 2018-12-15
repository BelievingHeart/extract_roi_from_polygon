//
// Created by afterburner on 18-12-15.
//
#include <opencv2/core/cvdef.h>          // for CV_PI
#include <cmath>                         // for atan2
#include <opencv2/core.hpp>              // for mean, multiply, eigen, trans...
#include "opencv2/highgui.hpp"           // for imshow, waitKey, destroyAllW...
#include "opencv2/imgproc.hpp"           // for Sobel, cvtColor, getRotation...

cv::Mat fix_rotation(cv::Mat &&image_segmented) {
  cv::Mat gaussian_out, image_bw;
  cv::cvtColor(image_segmented, image_bw, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(image_bw, gaussian_out, {3, 3}, 3);
  cv::Mat Ix, Iy;
  cv::Sobel(gaussian_out, Ix, CV_32F, 1, 0);
  cv::Sobel(gaussian_out, Iy, CV_32F, 0, 1);
  cv::Mat Ixx, Iyy, Ixy;
  cv::multiply(Ix, Ix, Ixx);
  cv::multiply(Ix, Iy, Ixy);
  cv::multiply(Iy, Iy, Iyy);
  cv::Scalar Ixx_mean = cv::mean(Ixx);
  cv::Scalar Iyy_mean = cv::mean(Iyy);
  cv::Scalar Ixy_mean = cv::mean(Ixy);
  const cv::Mat covariance_matrix =
          (cv::Mat_<float>(2, 2) << static_cast<float>(Ixx_mean[0]),
                  static_cast<float>(Ixy_mean[0]), static_cast<float>(Ixy_mean[0]),
                  static_cast<float>(Iyy_mean[0]));
  cv::Mat eigen_values, eigen_vectors;
  cv::eigen(covariance_matrix, eigen_values, eigen_vectors);
  cv::Mat eigen_vectors_T;
  cv::transpose(eigen_vectors, eigen_vectors_T);
  double angle_1 =
          atan2(eigen_vectors_T.at<float>(0, 0), eigen_vectors_T.at<float>(1, 0)) *
          180.0 / CV_PI;
  double angle_2 =
          atan2(eigen_vectors_T.at<float>(1, 0), eigen_vectors_T.at<float>(0, 0)) *
          180.0 / CV_PI;

  // get the area center
  const auto m = cv::moments(image_bw, true);
  const auto center_x = static_cast<float>(m.m10 / m.m00);
  const auto center_y = static_cast<float>(m.m01 / m.m00);

  // get rotation matrix
  const auto rotation_matrix_1 =
          cv::getRotationMatrix2D({center_x, center_y}, angle_1, 1);
  const auto rotation_matrix_2 =
          cv::getRotationMatrix2D({center_x, center_y}, angle_2, 1);

  // warp
  cv::Mat rotated_1, rotated_2;
  cv::warpAffine(image_segmented, rotated_1, rotation_matrix_1, {});
  cv::warpAffine(image_segmented, rotated_2, rotation_matrix_2, {});

  cv::Mat ret;
  // if
  int key = 0;
  cv::imshow("If this looks right, press 'a'", rotated_1);
  key = cv::waitKey(0);
  cv::imshow("If this looks right, press 'b'", rotated_2);
  key = cv::waitKey(0);
  if (static_cast<char>(key) == 'a') {
    ret = rotated_1;
  } else {
    ret = rotated_2;
  }

  cv::destroyAllWindows();
  cv::cvtColor(ret, image_bw, cv::COLOR_BGR2GRAY);
  const auto bounding_rect = cv::boundingRect(image_bw);
  ret = cv::Mat(ret, bounding_rect);
  return ret;
}