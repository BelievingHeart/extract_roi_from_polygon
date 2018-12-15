//
// Created by afterburner on 18-12-15.
//
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath>
#include <fmt/printf.h>

cv::Mat segment(const cv::Mat &contour_image, cv::Mat &&src) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(contour_image, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  cv::Mat dist_map(contour_image.size(), CV_32FC1);
  for (int r = 0; r < dist_map.rows; r++) {
    for (int c = 0; c < dist_map.cols; c++) {
      dist_map.at<float>(r, c) =
              cv::pointPolygonTest(contours[0], cv::Point2f(c, r), false);
    }
  }

  const auto area_to_discard = dist_map < 0;
  src.setTo(0, area_to_discard);
  return src;
}