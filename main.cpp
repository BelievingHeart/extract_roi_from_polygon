
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath>
#include <fmt/printf.h>
#include <iostream>

static cv::Mat canvas_color, canvas_bw, src;
static std::vector<cv::Point2i> vertices;
static const char *winName = "left click 4 points and right click to close them";

void click_and_draw(int event, int x, int y, int flags, void *usr_data);
extern cv::Mat segment(const cv::Mat &contour_image, cv::Mat &&src);
extern cv::Mat fix_rotation(cv::Mat &&image_segmented);
extern cv::Mat fit_bounding_box(cv::Mat &&contour_image, cv::Mat &&src);

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

  // rotation rectification and slice sub-region
  if (retify_rotation) {
    auto image_segmented = segment(canvas_bw, std::move(src));
    const cv::Mat sub_region = fix_rotation(std::move(image_segmented));
    cv::imshow("Final result", sub_region);
    cv::waitKey(0);
  } else {
    const cv::Mat sub_region =
        fit_bounding_box(std::move(canvas_bw), std::move(src));
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
