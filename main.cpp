
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath>
#include <fmt/printf.h>
#include <iostream>

static cv::Mat canvas_color, canvas_bw, src;
static std::vector<cv::Point2i> vertices;
static const char *winName = "src image";

void click_and_draw(int event, int x, int y, int flags, void *usr_data);
cv::Mat segment(cv::Mat &&contour_image);
cv::Mat fix_rotation(cv::Mat &&image_segmented);

int main(int argc, char **argv) {
  const cv::CommandLineParser parser(
      argc, argv,
      "{help ? h ||}"
      "{@input | /home/afterburner/Downloads/lvlian/lvlian_8.jpeg | input "
      "image}"
      "{scale_factor| 0.5| image scale factor}"
      "{rotate | true | rotation rectification }");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  const bool retify_rotation = parser.get<bool>("rotate");
  const auto image_path = parser.get<std::string>("@input");
  src = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src.empty()) {
    fmt::print("Error reading image <{}>\n", image_path);
    return -1;
  }
  const double scale_factor = parser.get<double>("scale_factor");
  cv::resize(src, src, {}, scale_factor, scale_factor, cv::INTER_LINEAR);

  // Outline contour
  cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(winName, click_and_draw, nullptr);
  cv::imshow(winName, src);
  cv::waitKey(0);
  cv::destroyWindow(winName);

  // Segment image according to contour
  auto image_segmented = segment(std::move(canvas_bw));
  cv::imshow("image_segmented", image_segmented);
  cv::waitKey(0);

  // rotation rectification and slice sub-region
  if (retify_rotation) {
    const cv::Mat rotation_rectified_image = fix_rotation(std::move(image_segmented));
    const auto bounding_rect = cv::boundingRect(rotation_rectified_image);
    const cv::Mat sub_region = cv::Mat(rotation_rectified_image, bounding_rect);
    cv::imshow("Final result", sub_region);
    cv::waitKey(0);
  }

  return 0;
}

void draw_lines_and_show(const std::vector<cv::Point2i> &vertices_,
                         cv::Mat &canvas, bool closed) {
  bool is_color = canvas.type() == CV_8UC3;
  int line_width = is_color ? 3 : 1;
  auto color = is_color ? cv::Scalar(255, 0, 0) : cv::Scalar::all(255);
  for (size_t i = 0; i < vertices_.size() - 1; i++) {
    cv::line(canvas, vertices_[i], vertices_[i + 1], color, line_width,
             cv::LINE_AA);
  }
  if (is_color) {
    for (const auto &pt : vertices_) {
      cv::circle(canvas, pt, 5, {0, 0, 255}, -1);
    }
  }
  if (closed) {
    cv::line(canvas, vertices_[0], vertices_.back(), color, line_width,
             cv::LINE_AA);
  }
  cv::imshow(winName, canvas);
}

void click_and_draw(int event, int x, int y, int flags, void *usr_data) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    vertices.emplace_back(x, y);
    canvas_color = src.clone();
    draw_lines_and_show(vertices, canvas_color, false);
  }
  if (event == cv::EVENT_RBUTTONDOWN) {
    canvas_bw = cv::Mat::zeros(src.size(), CV_8UC1);
    draw_lines_and_show(vertices, canvas_bw, true);
  }
}

cv::Mat segment(cv::Mat &&contour_image) {
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
  cv::Mat src_bw;
  cv::cvtColor(src, src_bw, cv::COLOR_BGR2GRAY);
  const auto area_to_discard = dist_map < 0;
  src_bw.setTo(0, area_to_discard);
  return src_bw;
}

cv::Mat fix_rotation(cv::Mat &&image_segmented) {
  cv::Mat gaussian_out;
  cv::GaussianBlur(image_segmented, gaussian_out, {3, 3}, 3);
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
  const auto m = cv::moments(image_segmented, true);
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
  return ret;
}
