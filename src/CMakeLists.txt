add_subdirectory(violib/src)

# Dependencies (optional, only used for example)
find_package(OpenCV REQUIRED)
find_package(depthai REQUIRED)


add_library(oak_utils SHARED "oak_utils.cpp")
target_link_libraries(oak_utils
  depthai::opencv 
  ${OpenCV_LIBS}
)

# Add source files
add_executable("${TARGET_NAME}"
    main.cpp
)

# Link with libraries
target_link_libraries(${TARGET_NAME}
    PUBLIC
        depthai::opencv
        ${OpenCV_LIBS} # optional, used for example
        violib
        oak_utils
)


## EXAMPLES =======

# Add source files
add_executable(feature_detector examples/feature_detector.cpp)
# Link with libraries
target_link_libraries(feature_detector PUBLIC depthai::opencv ${OpenCV_LIBS})

# Add source files
add_executable(calibration_reader examples/calibration_reader.cpp)
# Link with libraries
target_link_libraries(calibration_reader PUBLIC depthai::opencv ${OpenCV_LIBS})

# Add source files
add_executable(save_images examples/save_images.cpp)
# Link with libraries
target_link_libraries(save_images PUBLIC depthai::opencv ${OpenCV_LIBS})
