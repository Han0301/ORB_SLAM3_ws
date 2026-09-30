cmake_minimum_required(VERSION 3.16)
project(orbslam3_ros2)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../local/lib/cmake/Pangolin")

set(ORB_SLAM3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ORB_SLAM3")
set(PANGOLIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../local")

# add_executable cpp for the nodes
add_executable(rgbd_node src/rgbd_node.cpp)
add_executable(dense_mapper src/dense_mapper.cpp)
add_executable(occupancy_mapper src/occupancy_mapper.cpp)
target_include_directories(rgbd_node PRIVATE
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${PANGOLIN_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../sysroot/usr/include
  ${OpenCV_INCLUDE_DIRS}
)
target_compile_definitions(rgbd_node PRIVATE HAVE_EPOXY)
ament_target_dependencies(rgbd_node
  rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs cv_bridge
  message_filters tf2_ros
)
ament_target_dependencies(dense_mapper
  rclcpp sensor_msgs geometry_msgs std_msgs std_srvs cv_bridge message_filters
)
ament_target_dependencies(occupancy_mapper
  rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs std_srvs cv_bridge
  message_filters tf2_ros
)
target_link_libraries(dense_mapper ${OpenCV_LIBS})
target_link_libraries(occupancy_mapper ${OpenCV_LIBS})
target_link_libraries(rgbd_node
  ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  Eigen3::Eigen
)
set_target_properties(rgbd_node PROPERTIES
  BUILD_RPATH "${ORB_SLAM3_ROOT}/lib;${PANGOLIN_ROOT}/lib"
  INSTALL_RPATH "${ORB_SLAM3_ROOT}/lib;${PANGOLIN_ROOT}/lib"
)

install(TARGETS rgbd_node dense_mapper occupancy_mapper DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})
ament_package()
