//
// Created by afterburner on 18-12-15.
//
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fmt/printf.h>

cv::Mat fit_bounding_box(cv::Mat &&contour_image, cv::Mat &&src) {
  std::vector<std::vector<cv::Point2i>> contours;
  cv::findContours(contour_image, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // get vertices of both
  std::vector<cv::Point2i> vertices;
  cv::approxPolyDP(contours[0], vertices, 5, true);
  const auto bounding_rect = cv::boundingRect(contour_image);
  std::vector<cv::Point2i> vertices_upright{
      {bounding_rect.x, bounding_rect.y},
      {bounding_rect.x, bounding_rect.y + bounding_rect.height},
      {bounding_rect.x + bounding_rect.width,
       bounding_rect.y + bounding_rect.height},
      {bounding_rect.x + bounding_rect.width, bounding_rect.y}};

  const auto vertices_int_to_float = [](const std::vector<cv::Point2i> vertices_2i){
    std::vector<cv::Point2f> vertices_2f;
      vertices_2f.reserve(4);
      for (const auto &pt2i:vertices_2i) {
        const auto x = static_cast<float>(pt2i.x);
        const auto y = static_cast<float>(pt2i.y);
        vertices_2f.emplace_back(x, y);
      }
      return vertices_2f;
  };
  const auto vertices_2f = vertices_int_to_float(vertices);
  const auto vertices_upright_2f = vertices_int_to_float(vertices_upright);

  // use the vertices above to determine transform relationship
  const auto transform = cv::getPerspectiveTransform(vertices_2f, vertices_upright_2f);
  // transform
  cv::warpPerspective(src, src, transform, src.size());
  return cv::Mat(src, bounding_rect);
}