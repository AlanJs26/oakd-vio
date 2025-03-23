add_subdirectory(violib/src)

# Dependencies (optional, only used for example)
find_package(OpenCV REQUIRED)
find_package(depthai REQUIRED)

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
)
