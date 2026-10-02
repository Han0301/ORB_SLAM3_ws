cmake_minimum_required(VERSION 3.8)
project(yolo_det)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(perception_interfaces REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc freetype)

set(TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../local/TensorRT-11.3")
set(TENSORRT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv-yolo-export/lib/python3.11/site-packages/tensorrt_libs")
set(CUDA_RUNTIME_ROOT "/home/h/miniconda3/envs/yolo/lib/python3.11/site-packages/nvidia/cuda_runtime")
set(CUDA_COMPILER_INCLUDE "/home/h/miniconda3/envs/yolo/lib/python3.11/site-packages/triton/backends/nvidia/include")

add_executable(yolo_node
  src/yolo_node.cpp
  src/tensorrt_detector.cpp
)

ament_target_dependencies(yolo_node
  rclcpp
  ament_index_cpp
  sensor_msgs
  cv_bridge
  image_transport
  perception_interfaces
)

target_include_directories(yolo_node PRIVATE
  src
)

target_include_directories(yolo_node SYSTEM PRIVATE
  ${TENSORRT_ROOT}/include
  ${CUDA_RUNTIME_ROOT}/include
  ${CUDA_COMPILER_INCLUDE}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_node
  ${TENSORRT_LIB_DIR}/libnvinfer.so.11
  ${CUDA_RUNTIME_ROOT}/lib/libcudart.so.12
  ${OpenCV_LIBS}
)

set_target_properties(yolo_node PROPERTIES
  BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_RUNTIME_ROOT}/lib"
  INSTALL_RPATH "${TENSORRT_LIB_DIR};${CUDA_RUNTIME_ROOT}/lib"
)

install(TARGETS yolo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Model artifacts are machine/version specific and are intentionally not kept in Git.
install(FILES models/dfine_s_obj2coco.engine
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  # Keep the project-wide Allman brace convention instead of ROS uncrustify's
  # conflicting default brace layout. Other configured linters remain active.
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
