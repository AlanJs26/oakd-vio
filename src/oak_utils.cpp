#include "oak_utils.hpp"

LR<cv::Mat> OAKStereoQueue::getLRFrames() {
  // Receive frames from device
  auto leftFrame = this->left->get<dai::ImgFrame>();
  auto rightFrame = this->right->get<dai::ImgFrame>();

  return {//
          .left = leftFrame->getCvFrame(),
          .right = rightFrame->getCvFrame()};
}

LR<std::vector<dai::TrackedFeature>> OAKStereoQueue::getTrackedFeatures() {
  return {//
          .left = this->leftFeatures->get<dai::TrackedFeatures>()->trackedFeatures,
          .right = this->rightFeatures->get<dai::TrackedFeatures>()->trackedFeatures};
}

OAKStereoQueue OAKStereoQueue::getOAKStereoQueue() {
  OAKStereoQueue stereoQueue;

  std::shared_ptr<dai::Pipeline> pipeline(new dai::Pipeline());
  stereoQueue.pipeline = pipeline;

  // Define sources and outputs
  auto monoLeft = pipeline->create<dai::node::MonoCamera>();
  auto monoRight = pipeline->create<dai::node::MonoCamera>();

  auto featureTrackerLeft = pipeline->create<dai::node::FeatureTracker>();
  auto featureTrackerRight = pipeline->create<dai::node::FeatureTracker>();

  // Define Links
  auto xoutLeft = pipeline->create<dai::node::XLinkOut>();
  auto xoutRight = pipeline->create<dai::node::XLinkOut>();

  auto xoutTrackedFeaturesLeft = pipeline->create<dai::node::XLinkOut>();
  auto xoutTrackedFeaturesRight = pipeline->create<dai::node::XLinkOut>();
  auto xinTrackedFeaturesConfig = pipeline->create<dai::node::XLinkIn>();

  // Name streams
  xoutLeft->setStreamName("left");
  xoutRight->setStreamName("right");

  xoutTrackedFeaturesLeft->setStreamName("trackedFeaturesLeft");
  xoutTrackedFeaturesRight->setStreamName("trackedFeaturesRight");
  xinTrackedFeaturesConfig->setStreamName("trackedFeaturesConfig");

  // Properties
  monoLeft->setCamera("left");
  monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
  monoRight->setCamera("right");
  monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

  // Disable optical flow
  featureTrackerLeft->initialConfig.setMotionEstimator(false);
  featureTrackerRight->initialConfig.setMotionEstimator(false);

  // Linking
  monoLeft->out.link(featureTrackerLeft->inputImage);
  featureTrackerLeft->passthroughInputImage.link(xoutLeft->input);
  featureTrackerLeft->outputFeatures.link(xoutTrackedFeaturesLeft->input);
  xinTrackedFeaturesConfig->out.link(featureTrackerLeft->inputConfig);

  monoRight->out.link(featureTrackerRight->inputImage);
  featureTrackerRight->passthroughInputImage.link(xoutRight->input);
  featureTrackerRight->outputFeatures.link(xoutTrackedFeaturesRight->input);
  xinTrackedFeaturesConfig->out.link(featureTrackerRight->inputConfig);

  auto featureTrackerConfig = featureTrackerRight->initialConfig.get();

  // Connect to device and start pipeline
  std::shared_ptr<dai::Device> device(new dai::Device(*stereoQueue.pipeline));
  stereoQueue.device = device;

  // Output queues used to receive the results
  stereoQueue.left = device->getOutputQueue("left", 8, false);
  stereoQueue.leftFeatures = device->getOutputQueue("trackedFeaturesLeft", 8, false);
  stereoQueue.right = device->getOutputQueue("right", 8, false);
  stereoQueue.rightFeatures = device->getOutputQueue("trackedFeaturesRight", 8, false);

  auto inputFeatureTrackerConfigQueue = device->getInputQueue("trackedFeaturesConfig");

  return stereoQueue;
}
