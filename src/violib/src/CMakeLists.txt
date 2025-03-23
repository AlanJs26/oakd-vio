find_package( OpenCV REQUIRED )

add_subdirectory(evaluate)

set(project_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/evaluate)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${project_INCLUDE_DIRS}
)

if(USE_CUDA)
    find_package(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
    add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

add_library(feature SHARED "feature.cpp")
add_library(bucket SHARED "bucket.cpp")
add_library(utils SHARED "utils.cpp")
add_library(visualOdometry SHARED "visualOdometry.cpp")
add_library(rgbd_standalone SHARED "rgbd_standalone.cpp")

add_library(Frame SHARED "Frame.cpp")

add_library(violib INTERFACE)
set_target_properties(violib PROPERTIES
  "INTERFACE_INCLUDE_DIRECTORIES" ${project_INCLUDE_DIRS}
)

target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils evaluate_odometry feature ${OpenCV_LIBS} )
target_link_libraries( visualOdometry utils bucket feature ${OpenCV_LIBS})


target_link_libraries( Frame ${OpenCV_LIBS} )

target_link_libraries(violib INTERFACE
  bucket feature utils visualOdometry Frame rgbd_standalone
)
