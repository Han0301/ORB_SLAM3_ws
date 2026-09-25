#!/usr/bin/env bash

export PATH="/opt/ros/jazzy/bin:/usr/bin:/bin:${PATH}"
source /opt/ros/jazzy/setup.bash
source /home/h/ORBSLAM_ws/install/setup.bash
export CMAKE_PREFIX_PATH="/home/h/ORBSLAM_ws/local:/home/h/ORBSLAM_ws/sysroot/opt/ros/jazzy:/home/h/ORBSLAM_ws/sysroot/usr:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/home/h/ORBSLAM_ws/third_party/ORB_SLAM3/lib:/home/h/ORBSLAM_ws/local/lib:/home/h/ORBSLAM_ws/sysroot/opt/ros/jazzy/lib:/home/h/ORBSLAM_ws/sysroot/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH:-}"
