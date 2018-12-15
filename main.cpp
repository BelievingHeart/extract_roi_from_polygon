#include <fmt/core.h>                    // for print
#include <opencv2/core/hal/interface.h>  // for CV_8UC1, CV_8UC3
#include <opencv2/core/mat.hpp>          // for Mat, MatSize
#include <opencv2/core/utility.hpp>      // for CommandLineParser
#include <opencv2/imgcodecs.hpp>         // for imread, IMREAD_COLOR
#include "opencv2/highgui.hpp"           // for imshow, waitKey, destroyWindow
#include "opencv2/imgproc.hpp"           // for line, circle, resize, LINE_AA

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
      "{rotate | false | true when dealing with things like book pages }");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  const auto rectify_rotation = parser.get<bool>("rotate");
  const auto image_path = parser.get<std::string>("@input");
  src = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src.empty()) {
    fmt::print("Error reading image <{}>\n", image_path);
    return -1;
  }
  const auto scale_factor = parser.get<double>("scale_factor");
  cv::resize(src, src, {}, scale_factor, scale_factor, cv::INTER_LINEAR);

  // Outline contour
  cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
  bool closed = false;
  cv::setMouseCallback(winName, click_and_draw, &closed);
  cv::imshow(winName, src);
  cv::waitKey(0);
  cv::destroyWindow(winName);

  // rotation rectification and slice sub-region
  if (rectify_rotation && closed) {
    auto image_segmented = segment(canvas_bw, std::move(src));
    const cv::Mat sub_region = fix_rotation(std::move(image_segmented));
    cv::imshow("Final result", sub_region);
    cv::waitKey(0);
  } else if(closed) {
    const cv::Mat sub_region =
        fit_bounding_box(std::move(canvas_bw), std::move(src));
        cv::imshow("Final result", sub_region);
        cv::waitKey(0);
  } else{
    fmt::print("Polygon not closed, please restart program.\n");
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
    bool *closed_ptr = static_cast<bool *>(usr_data);
    *closed_ptr = true;
  }
}
