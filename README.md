# Gemini 2L + ORB-SLAM3 RGB-D-Inertial + ROS 2

适用于：

- Ubuntu 24.04
- ROS 2 Jazzy
- Orbbec Gemini 2 L
- ORB-SLAM3 RGB-D-Inertial
- Orbbec SDK v2
- 相机固件 1.5.02

相机驱动、SLAM、稠密建图和可视化模块分别启动，便于单独调试。

## 快速开始

进入工作区并加载环境：

```bash
cd /home/h/ORBSLAM_ws
source setup_env.bash
```

按照下面的顺序，在不同终端中分别启动所需模块。

### 1. 启动 Gemini 2L

```bash
ros2 run orbslam3_ros2 gemini2l_camera
```

### 2. 启动无界面 SLAM

```bash
ros2 run orbslam3_ros2 gemini2l_slam
```

适合长期运行、记录数据和后台建图。

### 3. 启动 ORB-SLAM3 原生前端

停止无界面 SLAM 后运行：

```bash
ros2 run orbslam3_ros2 gemini2l_slam_viewer
```

Pangolin 前端可以显示：

- 当前跟踪图像；
- ORB 特征；
- 稀疏地图点；
- 关键帧；
- 共视图；
- 相机轨迹。

`gemini2l_slam` 与 `gemini2l_slam_viewer` 不要同时运行。

### 4. 启动全局稠密建图

```bash
ros2 run orbslam3_ros2 dense_mapper
```

全局彩色点云发布到：

```text
/global_dense_map
```

点云每 10 秒自动保存到：

```text
/home/h/ORBSLAM_ws/maps/latest_dense_map.ply
```

需要立即保存时执行：

```bash
ros2 service call /dense_mapper/save std_srvs/srv/Trigger '{}'
```

清空当前地图：

```bash
ros2 service call /dense_mapper/clear std_srvs/srv/Empty '{}'
```

清空前会先保存当前点云。

### 5. 启动 RViz

```bash
ros2 run orbslam3_ros2 gemini2l_rviz
```

主要显示内容：

- `/global_dense_map`：全局彩色稠密点云；
- `/orb_slam3/map_points`：ORB-SLAM3 稀疏地图点；
- `/orb_slam3/path`：相机轨迹；
- `/orb_slam3/odometry`：当前位姿；
- `orb_map -> orb_camera`：相机 TF。

## 更新日志
260925
本次更新实现了 ORB-SLAM3 RGB-D-Inertial 兼容补丁的应用与基础运行状态检查，并基本实现了 Gemini 2L RGB-D-Inertial 数据接入、全局稠密地图构建及可视化支持
问题： **漂移、墙面增厚和少量离群点。**。

##
