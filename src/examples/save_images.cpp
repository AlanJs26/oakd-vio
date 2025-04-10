#include <depthai-shared/common/CameraBoardSocket.hpp>
#include <depthai-shared/datatype/RawTrackedFeatures.hpp>
#include <depthai-shared/properties/ColorCameraProperties.hpp>
#include <depthai/device/DataQueue.hpp>
#include <depthai/pipeline/Pipeline.hpp>
#include <depthai/pipeline/datatype/ImgFrame.hpp>
#include <depthai/pipeline/node/ColorCamera.hpp>
#include <depthai/pipeline/node/MonoCamera.hpp>
#include <depthai/pipeline/node/XLinkOut.hpp>

#include <depthai/depthai.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <vector>

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

template <typename T> struct LR {
  T left;
  T right;
};

struct OAKStereoQueue {
  std::shared_ptr<dai::DataOutputQueue> left;
  std::shared_ptr<dai::DataOutputQueue> right;
  std::shared_ptr<dai::DataOutputQueue> color;
  std::shared_ptr<dai::DataInputQueue> control;

  std::shared_ptr<dai::Device> device;
  std::shared_ptr<dai::Pipeline> pipeline;

  void getLRFrames(cv::Mat &left, cv::Mat &right);
  LR<cv::Mat> getLRFrames();
  static OAKStereoQueue getOAKStereoQueue();
};

LR<cv::Mat> OAKStereoQueue::getLRFrames() {
  cv::Mat leftFrame, rightFrame;
  this->getLRFrames(leftFrame, rightFrame);

  return {leftFrame, rightFrame};
}
void OAKStereoQueue::getLRFrames(cv::Mat &left, cv::Mat &right) {
  // Receive frames from device
  auto leftFrame = this->left->get<dai::ImgFrame>();
  auto rightFrame = this->right->get<dai::ImgFrame>();

  left = leftFrame->getCvFrame();
  right = rightFrame->getCvFrame();
}

OAKStereoQueue OAKStereoQueue::getOAKStereoQueue() {
  OAKStereoQueue stereoQueue;

  std::shared_ptr<dai::Pipeline> pipeline(new dai::Pipeline());
  stereoQueue.pipeline = pipeline;

  // Define sources and outputs
  auto monoLeft = pipeline->create<dai::node::MonoCamera>();
  auto monoRight = pipeline->create<dai::node::MonoCamera>();
  auto rgbCamera = pipeline->create<dai::node::ColorCamera>();

  // dai::CameraControl ctrl;
  // ctrl.setAutoExposureEnable();
  // controlQueue->send(ctrl);

  // Define Links
  auto xoutLeft = pipeline->create<dai::node::XLinkOut>();
  auto xoutRight = pipeline->create<dai::node::XLinkOut>();
  auto xoutRgb = pipeline->create<dai::node::XLinkOut>();

  auto xControl = pipeline->create<dai::node::XLinkIn>();

  // Name streams
  xoutLeft->setStreamName("left");
  xoutRight->setStreamName("right");
  xControl->setStreamName("control");
  xoutRgb->setStreamName("rgb");

  // Properties
  monoLeft->setCamera("left");
  monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
  monoRight->setCamera("right");
  monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

  rgbCamera->setPreviewSize(640, 480);
  rgbCamera->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  rgbCamera->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
  rgbCamera->setInterleaved(false);
  rgbCamera->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);

  // Linking
  monoRight->out.link(xoutRight->input);
  monoLeft->out.link(xoutLeft->input);
  xControl->out.link(monoLeft->inputControl);
  xControl->out.link(monoRight->inputControl);
  rgbCamera->preview.link(xoutRgb->input);

  // Connect to device and start pipeline
  std::shared_ptr<dai::Device> device(new dai::Device(*stereoQueue.pipeline));
  stereoQueue.device = device;

  // Output queues used to receive the results
  stereoQueue.left = device->getOutputQueue("left", 2, false);
  stereoQueue.right = device->getOutputQueue("right", 2, false);
  stereoQueue.color = device->getOutputQueue("rgb", 2, false);
  stereoQueue.control = device->getInputQueue(xControl->getStreamName());

  return stereoQueue;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: ./save_images [path_to_folder] [exposure_time] [iso]" << std::endl;
    return 1;
  }

  auto folder_path = argv[1];
  int exposure_time = std::stoi(argv[2]);
  int iso = std::stoi(argv[3]);

  auto stereoQueue = OAKStereoQueue::getOAKStereoQueue();

  // printf("Manual Exposure => time_us: %d  iso: %d\n", exposure_time, iso);
  // dai::CameraControl ctrl;
  // ctrl.setManualExposure(exposure_time, iso);
  // stereoQueue.control->send(ctrl);

  long count = 0;
  bool ready = false;

  std::cout << std::format("Pressione 'r' para começar a gravacao em {}", folder_path) << std::endl;
  std::cout << std::format("Pressione 'q' para parar o programa") << std::endl;

  while (true) {
    auto [leftFrame, rightFrame] = stereoQueue.getLRFrames();
    auto colorFrame = stereoQueue.color->get<dai::ImgFrame>()->getCvFrame();

    std::ostringstream pathLeft, pathRight, pathColor;

    pathLeft << folder_path << "/left/" << std::setw(6) << std::setfill('0') << count << ".jpg";
    pathRight << folder_path << "/right/" << std::setw(6) << std::setfill('0') << count << ".jpg";
    pathColor << folder_path << "/color/" << std::setw(6) << std::setfill('0') << count << ".jpg";

    cv::imshow("color", colorFrame);

    if (ready) {
      std::cout << pathColor.str() << std::endl;
      cv::imwrite(pathLeft.str(), leftFrame);
      cv::imwrite(pathRight.str(), rightFrame);
      cv::imwrite(pathColor.str(), colorFrame);
    }

    int key = cv::waitKey(1);

    if (key == 'q' || key == 'Q') {
      return 0;
    }
    if (key == 'r') {
      ready = true;
      std::cout << "Pronto!" << std::endl;
    }
    count++;
  }

  return 0;
}
