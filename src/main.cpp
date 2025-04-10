#include <depthai-shared/datatype/RawTrackedFeatures.hpp>
#include <depthai/device/DataQueue.hpp>
#include <depthai/pipeline/Pipeline.hpp>
#include <depthai/pipeline/node/MonoCamera.hpp>
#include <depthai/pipeline/node/XLinkOut.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <vector>

// include depthai library
#include <depthai/depthai.hpp>

// include opencv library (Optional, used only for the following example)
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <unistd.h>

#include "violib/src/evaluate/evaluate_odometry.h"
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

void filterCommonFeatures(std::vector<dai::TrackedFeature> &vec1, std::vector<dai::TrackedFeature> &vec2) {
  // Criar conjuntos de IDs
  std::unordered_set<int> ids1, ids2;
  for (const auto &f : vec1)
    ids1.insert(f.id);
  for (const auto &f : vec2)
    ids2.insert(f.id);

  // Encontrar interseção dos IDs
  std::unordered_set<int> common_ids;
  for (int id : ids1) {
    if (ids2.find(id) != ids2.end()) {
      common_ids.insert(id);
    }
  }

  // Remover elementos que não estão na interseção
  auto remove_unmatched = [&](const dai::TrackedFeature &f) { return common_ids.find(f.id) == common_ids.end(); };

  vec1.erase(std::remove_if(vec1.begin(), vec1.end(), remove_unmatched), vec1.end());
  vec2.erase(std::remove_if(vec2.begin(), vec2.end(), remove_unmatched), vec2.end());
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
  if (argc < 2 || argc > 4) {
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

  // clock_t t_A, t_B;
  // stereoQueue = OAKStereoQueue::getOAKStereoQueue();
  // while (true) {
  //   t_A = clock();
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
  //
  //   t_B = clock();
  //   float frame_time = 1000 * (double)(t_B - t_A) / CLOCKS_PER_SEC;
  //   float fps = 1000 / frame_time;
  //   cout << "[Info] frame times (ms): " << frame_time << endl;
  //   cout << "[Info] FPS: " << fps << endl;
  //
  //   if (key == 'q' || key == 'Q') {
  //     return 0;
  //   }
  // }
  // return 0;

  std::vector<Matrix> pose_matrix_gt;
  bool display_ground_truth = false;
  if (argc == 4) {
    display_ground_truth = true;
    cerr << "Display ground truth trajectory" << endl;
    // load ground truth pose
    string filename_pose = string(argv[3]);
    pose_matrix_gt = loadPoses(filename_pose);
  }
  if (argc >= 3) {
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
  if (bf == 0) {
    float baseline = fSettings["Camera.baseline"];
    bf = -baseline * fx;
    cout << "Baseline: " << baseline << "  bf: " << bf << endl;
  }

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
  cv::Mat trajectory = cv::Mat::zeros(1000, 1000, CV_8UC3);
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

  // LR<std::vector<dai::TrackedFeature>> features_t0, features_t1;
  std::vector<dai::TrackedFeature> features;

  // -----------------------------------------
  // Run visual odometry
  // -----------------------------------------

  for (int frame_id = init_frame_id + 1; use_oakd ? true : frame_id < 9000; frame_id++) {

    std::cout << std::endl << "frame id " << frame_id << std::endl;

    // ------------
    // Load images
    // ------------
    cv::Mat imageRight_t1, imageLeft_t1;
    if (use_oakd) {
      features = stereoQueue.leftFeatures->get<dai::TrackedFeatures>()->trackedFeatures;
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

    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;

    // cout << "pointsLeft_t0 -- size: " << pointsLeft_t0.size() << endl;

    // ----------------------------
    // Feature detection using FAST
    // ----------------------------

    if (currentVOFeatures.size() < 2000) {
      // append new features with old features
      if (use_oakd) {
        for (auto &feature : features) {
          currentVOFeatures.points.push_back(cv::Point2f(feature.position.x, feature.position.y));
          currentVOFeatures.ages.push_back(0);
        }
      } else {
        appendNewFeatures(imageLeft_t0, currentVOFeatures);
      }

      std::cout << "Current feature set size: " << currentVOFeatures.points.size() << std::endl;
    }

    if (currentVOFeatures.points.size() > 0) {
      matchingFeatures(imageLeft_t0, imageRight_t0,   // image at previous iteration
                       imageLeft_t1, imageRight_t1,   // image at current iteration
                       currentVOFeatures,             // Features
                       pointsLeft_t0, pointsRight_t0, // points at previous iteration
                       pointsLeft_t1, pointsRight_t1  // points at current iteration
      );
    }

    // ---------------------
    // Triangulate 3D Points
    // ---------------------

    imageLeft_t0 = imageLeft_t1;
    imageRight_t0 = imageRight_t1;
    if (pointsLeft_t0.size() < 4 || currentVOFeatures.points.size() == 0) {
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
        if (integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation)) {
          // imageLeft_t0 = imageLeft_t1;
          // imageRight_t0 = imageRight_t1;
        }
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

    cv::Mat xyz = frame_pose.col(3).clone();
    std::cout << xyz.at<double>(0) << ", " << xyz.at<double>(1) << ", " << xyz.at<double>(2) << std::endl;

    cv::putText(imageLeft_t1, cv::format("%f, %f, %f", xyz.at<double>(0), xyz.at<double>(1), xyz.at<double>(2)), cv::Point2d(20, 100), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 255, 0), 4);
    displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

    if (display_ground_truth) {
      display(frame_id, trajectory, xyz, pose_matrix_gt, fps, true);
    } else {
      display(frame_id, trajectory, xyz, fps);
    }
    cv::waitKey(1);
  }

  return 0;
}
