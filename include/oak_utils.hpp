#include <depthai/device/DataQueue.hpp>
#include <depthai/pipeline/Pipeline.hpp>
#include <depthai/pipeline/node/MonoCamera.hpp>
#include <depthai/pipeline/node/XLinkOut.hpp>

// include depthai library
#include <depthai/depthai.hpp>

// include opencv library (Optional, used only for the following example)
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

template <typename T> struct LR {
  T left;
  T right;
};

struct OAKStereoQueue {
  std::shared_ptr<dai::DataOutputQueue> left;
  std::shared_ptr<dai::DataOutputQueue> right;
  std::shared_ptr<dai::DataOutputQueue> leftFeatures;
  std::shared_ptr<dai::DataOutputQueue> rightFeatures;

  std::shared_ptr<dai::Device> device;
  std::shared_ptr<dai::Pipeline> pipeline;

  LR<std::vector<dai::TrackedFeature>> getTrackedFeatures();
  void getLRFrames(cv::Mat &left, cv::Mat &right);
  LR<cv::Mat> getLRFrames();
  static OAKStereoQueue getOAKStereoQueue();
};
