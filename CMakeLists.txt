cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# Create a project
set(TARGET_NAME oakdvio)
project(${TARGET_NAME})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_COLOR_DIAGNOSTICS ON)

include_directories(include)

add_subdirectory(src)

# Set some errors
if(NOT MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
endif()

# Set compiler features (c++14)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

# Windows - Add runtime dependencies
if(WIN32)
  if(CMAKE_VERSION VERSION_LESS "3.21")
    message(WARNING "CMake version less than 3.21 - automatic DLL handling not available. Make sure to copy required DLLs to the same folder as .exe")
  else()
    # TARGET_RUNTIME_DLLS generator expression available since CMake 3.21
    set(depthai_dll_libraries "$<TARGET_RUNTIME_DLLS:${TARGET_NAME}>")
    # Copy the required dlls
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${depthai_dll_libraries} $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMAND_EXPAND_LISTS
        )
  endif()
endif()
