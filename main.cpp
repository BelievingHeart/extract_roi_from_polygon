
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fmt/printf.h>

cv::Mat canvas_color, canvas_bw, src;
std::vector<cv::Point2i> vertices;
const char *winName = "src image";

void click_and_draw(int event, int x, int y, int flags, void *usr_data);
cv::Mat segment(cv::Mat &&contour_image);

int main(int argc, char **argv) {
  cv::CommandLineParser parser(
      argc, argv,
      "{@input | /home/afterburner/Pictures/wanglaoji.jpg | input "
      "image}");
  const auto image_path = parser.get<std::string>("@input");
  src = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src.empty()) {
    fmt::print("Error reading image <{}>\n", image_path);
    return -1;
  }

  cv::resize(src, src, {}, 0.2, 0.2, cv::INTER_LINEAR);

  // Outline contour
  cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(winName, click_and_draw, nullptr);
  cv::imshow(winName, src);
  cv::waitKey(0);

  // Segment image according to contour
  const auto image_segmented = segment(std::move(canvas_bw));
  cv::imshow(winName, image_segmented);
  cv::waitKey(0);
}

void draw_lines_and_show(std::vector<cv::Point2i> &vertices_, cv::Mat &canvas,
                         bool closed) {
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
    fmt::print("ROI selected, press any key to continue...\n");
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
