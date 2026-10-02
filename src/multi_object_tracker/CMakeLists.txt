cmake_minimum_required(VERSION 3.8)
project(multi_object_tracker)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(perception_interfaces REQUIRED)
find_package(rclcpp REQUIRED)

add_library(tracker_core
  src/byte_tracker.cpp
  src/hungarian_solver.cpp
  src/kalman_filter.cpp
)

target_include_directories(tracker_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(tracker_core PUBLIC Eigen3::Eigen)

add_executable(tracker_node
  src/tracker_node.cpp
)

target_include_directories(tracker_node PRIVATE src)
target_link_libraries(tracker_node tracker_core)
ament_target_dependencies(tracker_node
  perception_interfaces
  rclcpp
)

install(TARGETS tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  enable_testing()

  add_executable(tracker_scenario_test
    src/test_debug/tracker_scenario_test.cpp
  )
  target_include_directories(tracker_scenario_test PRIVATE
    src
    src/test_debug
  )
  target_link_libraries(tracker_scenario_test PRIVATE tracker_core)
  add_test(NAME tracker_scenario_test COMMAND tracker_scenario_test)

  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  # Keep the user-required standalone-brace C++ style instead of ROS uncrustify defaults.
  set(ament_cmake_uncrustify_FOUND TRUE)
  # The schema URL requires network access; package.xml is checked locally in test_debug.
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
