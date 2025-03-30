#include <depthai-shared/datatype/RawTrackedFeatures.hpp>
#include <depthai/device/DataQueue.hpp>
#include <depthai/pipeline/Pipeline.hpp>
#include <depthai/pipeline/node/MonoCamera.hpp>
#include <depthai/pipeline/node/XLinkOut.hpp>
#include <iostream>

#include <algorithm>
#include <iterator>

// include depthai library
#include <depthai/depthai.hpp>

// include opencv library (Optional, used only for the following example)
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <unistd.h>

#include "violib/src/visualOdometry.h"

#include "oak_utils.hpp"

static void drawFeatures(cv::Mat &frame, std::vector<dai::TrackedFeature> &features) {
  static const auto pointColor = cv::Scalar(0, 0, 255);
  static const int circleRadius = 2;
  for (auto &feature : features) {
    cv::circle(frame, cv::Point(feature.position.x, feature.position.y), circleRadius, pointColor, -1, cv::LINE_AA, 0);
  }
}

void features2points(std::vector<dai::TrackedFeature> &features, std::vector<cv::Point2f> &points) {
  points.clear();
  for (auto feature : features) {
    points.push_back(cv::Point2f(feature.position.x, feature.position.y));
  }
}

void normalizeFeatures(LR<std::vector<dai::TrackedFeature>> &features_t0, LR<std::vector<dai::TrackedFeature>> &features_t1) {
  struct iterStruct {
    std::vector<dai::TrackedFeature> feature;
    int i;

    std::vector<dai::TrackedFeature>::iterator current() { return this->feature.begin() + this->i; }
  };

  std::array<iterStruct, 4> iterators = {
      iterStruct{features_t0.left, 0},
      iterStruct{features_t0.right, 0},
      iterStruct{features_t1.left, 0},
      iterStruct{features_t1.right, 0},
  };

  // Ordena em features em ordem crescente
  for (iterStruct &it : iterators) {
    std::sort(it.feature.begin(), it.feature.end(), [](dai::TrackedFeature &a, dai::TrackedFeature &b) { return a.id < b.id; });
  }

  while (true) {
    // Verifica se existe um iterator que acabou
    auto ended_iter = std::find_if(iterators.begin(), iterators.end(), [](iterStruct a) { return a.current() == a.feature.end(); });
    if (ended_iter != iterators.end()) {
      // retrocede um item e termina o while
      for (iterStruct &it : iterators) {
        it.i--;
      }
      break;
    }

    // verifica se o feature.id dos iterators são todos iguais
    auto equal_id_iter = std::adjacent_find(iterators.begin(), iterators.end(), [](iterStruct a, iterStruct b) { return a.current()->id != b.current()->id; });
    if (equal_id_iter == iterators.end()) {
      for (auto &it : iterators) {
        it.i++;
      }
      continue;
    }

    iterStruct *max_id_iter =
        std::max_element(std::begin(iterators), std::end(iterators), [](iterStruct a, iterStruct b) { return a.current()->id < b.current()->id; });

    for (iterStruct &it : iterators) {
      if (&it != max_id_iter) {
        it.feature.erase(it.current());
      }
    }
  }

  for (iterStruct &it : iterators) {
    it.feature = std::vector<dai::TrackedFeature>(it.feature.begin(), it.current());
  }
}

int main(int argc, char **argv) {
  using namespace std;

#if USE_CUDA
  printf("CUDA Enabled\n");
#endif
  // -----------------------------------------
  // Load dataset images and calibration parameters
  // -----------------------------------------
  bool use_oakd = false;
  if (argc < 2 || argc > 3) {
    cerr << "Usage (kitti dataset): ./run [path_to_sequence] "
            "[path_to_calibration]"
         << endl;
    cerr << "Usage (OAK-D):         ./run [path_to_calibration]" << endl;
    return 1;
  }

  // Sequence
  string filepath;
  // Settings file (Camera calibration)
  string strSettingPath;
  // OAK-D cameras queues
  OAKStereoQueue stereoQueue;

  // stereoQueue = OAKStereoQueue::getOAKStereoQueue();
  // while (true) {
  //   cv::Mat leftFrameColor, rightFrameColor;
  //
  //   auto [leftFrame, rightFrame] = stereoQueue.getLRFrames();
  //   auto features = stereoQueue.getTrackedFeatures();
  //
  //   cv::cvtColor(leftFrame, leftFrameColor, cv::COLOR_GRAY2BGR);
  //
  //   drawFeatures(leftFrameColor, features.left);
  //   cv::imshow("left", leftFrameColor);
  //
  //   int key = cv::waitKey(1);
  //   if (key == 'q' || key == 'Q') {
  //     return 0;
  //   }
  // }
  // return 0;

  if (argc == 3) {
    cout << "Using KITTI dataset" << endl;
    use_oakd = false;

    filepath = string(argv[1]);
    strSettingPath = string(argv[2]);
    cout << "Filepath: " << filepath << endl;
    cout << "Calibration Filepath: " << strSettingPath << endl;
  } else {
    cout << "Using OAK-D camera" << endl;
    use_oakd = true;

    stereoQueue = OAKStereoQueue::getOAKStereoQueue();

    strSettingPath = string(argv[1]);
    cout << "Calibration Filepath: " << strSettingPath << endl;
  }

  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];
  float bf = fSettings["Camera.bf"];

  cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
  cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0, 0., 1., 0.);
  cout << "P_left: " << endl << projMatrl << endl;
  cout << "P_right: " << endl << projMatrr << endl;

  // -----------------------------------------
  // Initialize variables
  // -----------------------------------------
  cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);

  cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);

  cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

  std::cout << "frame_pose:" << endl << frame_pose << std::endl;
  cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
  FeatureSet currentVOFeatures;
  cv::Mat points4D, points3D;
  int init_frame_id = 0;

  // ------------------------
  // Load first images
  // ------------------------
  cv::Mat imageRight_t0, imageLeft_t0;
  if (use_oakd) {
    stereoQueue.getLRFrames(imageLeft_t0, imageRight_t0);
  } else {
    cv::Mat imageLeft_t0_color;
    loadImageLeft(imageLeft_t0_color, imageLeft_t0, init_frame_id, filepath);

    cv::Mat imageRight_t0_color;
    loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);
  }
  cout << "Rows: " << imageLeft_t0.rows << " Columns: " << imageLeft_t0.cols << endl;
  clock_t t_a, t_b;

  // ----------------
  // Extract first features
  // ----------------

  LR<std::vector<dai::TrackedFeature>> features_t0, features_t1;
  features_t1 = stereoQueue.getTrackedFeatures();

  // -----------------------------------------
  // Run visual odometry
  // -----------------------------------------

  for (int frame_id = init_frame_id + 1; use_oakd ? true : frame_id < 9000; frame_id++) {

    std::cout << std::endl << "frame id " << frame_id << std::endl;

    features_t0 = features_t1;

    // ------------
    // Load images
    // ------------
    cv::Mat imageRight_t1, imageLeft_t1;
    if (use_oakd) {
      stereoQueue.getLRFrames(imageLeft_t1, imageRight_t1);
    } else {
      cv::Mat imageLeft_t1_color;
      loadImageLeft(imageLeft_t1_color, imageLeft_t1, frame_id, filepath);
      cv::Mat imageRight_t1_color;
      loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath);
    }

    t_a = clock();

    // ----------------
    // Extract Features
    // ----------------

    features_t1 = stereoQueue.getTrackedFeatures();

    normalizeFeatures(features_t0, features_t1);

    auto feature_vectors = {features_t0.left, features_t0.right, features_t1.left, features_t1.right};
    auto equal_sizes_iter = std::adjacent_find(feature_vectors.begin(), feature_vectors.end(),
                                               [](std::vector<dai::TrackedFeature> a, std::vector<dai::TrackedFeature> b) { return a.size() == a.size(); });
    assert(equal_sizes_iter != feature_vectors.end());

    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;

    features2points(features_t0.left, pointsLeft_t0);
    features2points(features_t0.right, pointsRight_t0);
    features2points(features_t1.left, pointsLeft_t1);
    features2points(features_t1.right, pointsRight_t1);

    // matchingFeatures(imageLeft_t0, imageRight_t0,   // image at previous iteration
    //                  imageLeft_t1, imageRight_t1,   // image at current iteration
    //                  currentVOFeatures,             // Features
    //                  pointsLeft_t0, pointsRight_t0, // points at previous iteration
    //                  pointsLeft_t1, pointsRight_t1  // points at current iteration
    // );

    // imageLeft_t0 = imageLeft_t1;
    // imageRight_t0 = imageRight_t1;

    // ---------------------
    // Triangulate 3D Points
    // ---------------------

    cout << "pointsLeft_t0 -- size: " << pointsLeft_t0.size() << endl;
    if (pointsLeft_t0.size() < 4) {
      cout << "Insufficient features found! Skiping iteration" << endl;
    } else {
      cv::Mat points3D_t0, points4D_t0;
      cv::triangulatePoints(projMatrl, projMatrr,          //
                            pointsLeft_t0, pointsRight_t0, //
                            points4D_t0);
      cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

      cv::Mat points3D_t1, points4D_t1;
      cv::triangulatePoints(projMatrl, projMatrr, pointsLeft_t1, pointsRight_t1, points4D_t1);
      cv::convertPointsFromHomogeneous(points4D_t1.t(), points3D_t1);

      // ---------------------
      // Tracking transfomation
      // ---------------------
      clock_t tic_gpu = clock();
      trackingFrame2Frame(projMatrl, projMatrr,         //
                          pointsLeft_t0, pointsLeft_t1, //
                          points3D_t0,                  //
                          rotation, translation, false);
      clock_t toc_gpu = clock();
      std::cerr << "tracking frame 2 frame: " << float(toc_gpu - tic_gpu) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

      points4D = points4D_t0;
      frame_pose.convertTo(frame_pose32, CV_32F);
      points4D = frame_pose32 * points4D;
      cv::convertPointsFromHomogeneous(points4D.t(), points3D);

      // ------------------------------------------------
      // Integrating and display
      // ------------------------------------------------

      cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);

      cv::Mat rigid_body_transformation;

      if (abs(rotation_euler[1]) < 0.1 && abs(rotation_euler[0]) < 0.1 && abs(rotation_euler[2]) < 0.1) {
        integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation);

      } else {

        std::cout << "Too large rotation" << std::endl;
      }
    }
    t_b = clock();
    float frame_time = 1000 * (double)(t_b - t_a) / CLOCKS_PER_SEC;
    float fps = 1000 / frame_time;
    cout << "[Info] frame times (ms): " << frame_time << endl;
    cout << "[Info] FPS: " << fps << endl;

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;
    // std::cout << "rotation: " << rotation_euler << std::endl;
    // std::cout << "translation: " << translation.t() << std::endl;
    // std::cout << "frame_pose" << frame_pose << std::endl;

    displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);
    cv::Mat xyz = frame_pose.col(3).clone();
    display(frame_id, trajectory, xyz, fps);
  }

  return 0;
}
