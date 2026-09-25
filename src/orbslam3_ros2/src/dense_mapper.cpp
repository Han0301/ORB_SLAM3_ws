#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/trigger.hpp>

struct VoxelKey {
  int x, y, z;
  bool operator==(const VoxelKey &o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelHash {
  std::size_t operator()(const VoxelKey &k) const {
    std::size_t h = std::hash<int>{}(k.x);
    h ^= std::hash<int>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};
struct VoxelValue { double x=0, y=0, z=0, r=0, g=0, b=0; uint32_t n=0; };

class DenseMapper : public rclcpp::Node {
 public:
  DenseMapper() : Node("dense_mapper") {
    voxel_ = declare_parameter("voxel_size", 0.04);
    stride_ = declare_parameter("pixel_stride", 4);
    min_depth_ = declare_parameter("min_depth", 0.25);
    max_depth_ = declare_parameter("max_depth", 6.0);
    publish_every_ = declare_parameter("publish_every_n_frames", 10);
    save_path_ = declare_parameter("save_path", "/home/h/ORBSLAM_ws/maps/latest_dense_map.ply");
    const auto autosave_seconds = declare_parameter("autosave_seconds", 10.0);
    const auto color = declare_parameter("color_topic", "/camera/color/image_raw");
    const auto depth = declare_parameter("depth_topic", "/camera/depth/image_raw");
    const auto pose = declare_parameter("pose_topic", "/orb_slam3/pose");
    const auto info = declare_parameter("camera_info_topic", "/camera/color/camera_info");
    const auto state = declare_parameter("tracking_state_topic", "/orb_slam3/tracking_state");
    const auto imu = declare_parameter("imu_initialized_topic", "/orb_slam3/imu_initialized");
    map_frame_ = declare_parameter("map_frame", "orb_map");

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info, rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
        std::lock_guard<std::mutex> l(calib_mutex_); fx_=m->k[0]; fy_=m->k[4]; cx_=m->k[2]; cy_=m->k[5];
      });
    state_sub_ = create_subscription<std_msgs::msg::Int32>(state, 10,
      [this](std_msgs::msg::Int32::ConstSharedPtr m) { tracking_ok_ = (m->data == 2); });
    imu_sub_ = create_subscription<std_msgs::msg::Bool>(imu, 10,
      [this](std_msgs::msg::Bool::ConstSharedPtr m) { imu_initialized_ = m->data; });
    clear_service_ = create_service<std_srvs::srv::Empty>("clear",
      [this](const std_srvs::srv::Empty::Request::SharedPtr,
             std_srvs::srv::Empty::Response::SharedPtr) { clearMap("manual request"); });
    save_service_ = create_service<std_srvs::srv::Trigger>("save",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        response->success = saveMap();
        response->message = response->success ? save_path_ : "dense map is empty or could not be written";
      });
    autosave_timer_ = create_wall_timer(
      std::chrono::duration<double>(autosave_seconds), [this]() {
        if (dirty_) saveMap();
      });
    color_sub_.subscribe(this, color, rmw_qos_profile_sensor_data);
    depth_sub_.subscribe(this, depth, rmw_qos_profile_sensor_data);
    pose_sub_.subscribe(this, pose, rmw_qos_profile_sensor_data);
    sync_ = std::make_shared<Sync>(Policy(30), color_sub_, depth_sub_, pose_sub_);
    sync_->registerCallback(std::bind(&DenseMapper::callback, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("global_dense_map", 1);
    RCLCPP_INFO(get_logger(), "Global dense mapper ready: voxel %.3f m, stride %d; saving to %s",
                voxel_, stride_, save_path_.c_str());
  }

  ~DenseMapper() override { saveMap(); }

 private:
  using Image = sensor_msgs::msg::Image;
  using Pose = geometry_msgs::msg::PoseStamped;
  using Policy = message_filters::sync_policies::ApproximateTime<Image, Image, Pose>;
  using Sync = message_filters::Synchronizer<Policy>;

  void callback(Image::ConstSharedPtr color_msg, Image::ConstSharedPtr depth_msg, Pose::ConstSharedPtr pose) {
    if (!tracking_ok_ || !imu_initialized_) {
      if (mapping_active_) clearMap("SLAM tracking lost or IMU not initialized");
      mapping_active_ = false;
      have_last_pose_ = false;
      return;
    }
    if (!mapping_active_) {
      clearMap("starting a new initialized SLAM map");
      mapping_active_ = true;
    }
    double fx, fy, cx, cy;
    { std::lock_guard<std::mutex> l(calib_mutex_); fx=fx_; fy=fy_; cx=cx_; cy=cy_; }
    if (fx <= 0 || fy <= 0) return;
    const cv::Mat color = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat depth;
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
        depth_msg->encoding == sensor_msgs::image_encodings::MONO16)
      cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image.convertTo(depth, CV_32F, 0.001);
    else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
      depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;
    else return;

    const auto &q=pose->pose.orientation; const auto &t=pose->pose.position;
    if (have_last_pose_) {
      const double dx=t.x-last_position_[0], dy=t.y-last_position_[1], dz=t.z-last_position_[2];
      const double translation=std::sqrt(dx*dx+dy*dy+dz*dz);
      const double dot=std::abs(q.x*last_orientation_[0]+q.y*last_orientation_[1]+
                                q.z*last_orientation_[2]+q.w*last_orientation_[3]);
      const double rotation=2.0*std::acos(std::min(1.0, dot));
      if (translation > 0.30 || rotation > 0.45) {
        clearMap("discontinuous SLAM pose");
        last_position_={t.x,t.y,t.z}; last_orientation_={q.x,q.y,q.z,q.w};
        have_last_pose_=true;
        return;
      }
    }
    last_position_={t.x,t.y,t.z}; last_orientation_={q.x,q.y,q.z,q.w}; have_last_pose_=true;
    const double xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    const double xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z, wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    const std::array<double,9> R={1-2*(yy+zz),2*(xy-wz),2*(xz+wy),2*(xy+wz),1-2*(xx+zz),2*(yz-wx),2*(xz-wy),2*(yz+wx),1-2*(xx+yy)};
    for (int v=0; v<depth.rows; v+=stride_) for (int u=0; u<depth.cols; u+=stride_) {
      const float z=depth.at<float>(v,u); if (!std::isfinite(z)||z<min_depth_||z>max_depth_) continue;
      const double x=(u-cx)*z/fx, y=(v-cy)*z/fy;
      const double X=R[0]*x+R[1]*y+R[2]*z+t.x, Y=R[3]*x+R[4]*y+R[5]*z+t.y, Z=R[6]*x+R[7]*y+R[8]*z+t.z;
      const VoxelKey key{(int)std::floor(X/voxel_),(int)std::floor(Y/voxel_),(int)std::floor(Z/voxel_)};
      auto &a=voxels_[key]; const auto bgr=color.at<cv::Vec3b>(v,u);
      a.x+=X; a.y+=Y; a.z+=Z; a.r+=bgr[2]; a.g+=bgr[1]; a.b+=bgr[0]; ++a.n;
    }
    dirty_=true;
    if (++frames_ % publish_every_ == 0) publish(pose->header.stamp);
  }

  bool saveMap() {
    if (voxels_.empty()) return false;
    try {
      const std::filesystem::path output(save_path_);
      if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
      const auto temporary = output.string() + ".tmp";
      std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
      if (!file) return false;
      file << "ply\nformat binary_little_endian 1.0\n"
           << "comment generated by orbslam3_ros2 dense_mapper\n"
           << "element vertex " << voxels_.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
      for (const auto &[key, value] : voxels_) {
        const double n = value.n;
        const std::array<float, 3> xyz = {
          static_cast<float>(value.x / n), static_cast<float>(value.y / n),
          static_cast<float>(value.z / n)};
        const std::array<uint8_t, 3> rgb = {
          static_cast<uint8_t>(value.r / n), static_cast<uint8_t>(value.g / n),
          static_cast<uint8_t>(value.b / n)};
        file.write(reinterpret_cast<const char *>(xyz.data()), sizeof(xyz));
        file.write(reinterpret_cast<const char *>(rgb.data()), sizeof(rgb));
      }
      file.close();
      if (!file) return false;
      std::filesystem::rename(temporary, output);
      dirty_=false;
      RCLCPP_INFO(get_logger(), "Saved dense map: %zu points -> %s", voxels_.size(), save_path_.c_str());
      return true;
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to save dense map: %s", e.what());
      return false;
    }
  }

  void publish(const builtin_interfaces::msg::Time &stamp) {
    sensor_msgs::msg::PointCloud2 out; out.header.stamp=stamp; out.header.frame_id=map_frame_;
    sensor_msgs::PointCloud2Modifier m(out); m.setPointCloud2FieldsByString(2,"xyz","rgb"); m.resize(voxels_.size());
    sensor_msgs::PointCloud2Iterator<float> x(out,"x"),y(out,"y"),z(out,"z");
    sensor_msgs::PointCloud2Iterator<uint8_t> r(out,"r"),g(out,"g"),b(out,"b");
    for (const auto &[k,a]:voxels_) { const double n=a.n; *x=a.x/n;*y=a.y/n;*z=a.z/n;*r=a.r/n;*g=a.g/n;*b=a.b/n;++x;++y;++z;++r;++g;++b; }
    cloud_pub_->publish(out);
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),5000,"Global dense map: %zu voxels",voxels_.size());
  }

  void clearMap(const char *reason) {
    if (!voxels_.empty()) RCLCPP_WARN(get_logger(), "Clearing dense map (%zu voxels): %s", voxels_.size(), reason);
    if (!voxels_.empty() && dirty_) saveMap();
    voxels_.clear(); frames_=0;
  }

  double voxel_,min_depth_,max_depth_,fx_=0,fy_=0,cx_=0,cy_=0; int stride_,publish_every_,frames_=0; std::string map_frame_,save_path_;
  std::mutex calib_mutex_; std::unordered_map<VoxelKey,VoxelValue,VoxelHash> voxels_;
  bool tracking_ok_=false, imu_initialized_=false, mapping_active_=false, have_last_pose_=false;
  bool dirty_=false;
  std::array<double,3> last_position_{}; std::array<double,4> last_orientation_{};
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr imu_sub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr clear_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
  rclcpp::TimerBase::SharedPtr autosave_timer_;
  message_filters::Subscriber<Image> color_sub_,depth_sub_; message_filters::Subscriber<Pose> pose_sub_;
  std::shared_ptr<Sync> sync_; rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

int main(int argc,char **argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<DenseMapper>());rclcpp::shutdown();return 0;}
