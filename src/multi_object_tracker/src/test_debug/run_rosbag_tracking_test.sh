#!/usr/bin/env bash

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <rosbag目录> [回放秒数]" >&2
  exit 2
fi

bag_path="$1"
duration_sec="${2:-12}"
workspace="/home/h/ORBSLAM_ws"
debug_dir="$workspace/src/multi_object_tracker/src/test_debug"
log_dir="$(mktemp -d /tmp/multi_object_tracker_bag.XXXXXX)"
process_ids=()

cleanup()
{
  for process_id in "${process_ids[@]}"; do
    kill "$process_id" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

source "$workspace/setup_env.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-87}"
export ROS_LOG_DIR="$log_dir/ros"
mkdir -p "$ROS_LOG_DIR"

ros2 run yolo_det yolo_node \
  --ros-args --params-file "$debug_dir/rosbag_yolo.yaml" \
  >"$log_dir/yolo.log" 2>&1 &
process_ids+=("$!")

ros2 launch multi_object_tracker multi_object_tracker.launch.py \
  >"$log_dir/tracker.log" 2>&1 &
process_ids+=("$!")

sleep 4

python3 "$debug_dir/tracking_topic_probe.py" \
  --timeout-sec "$((duration_sec + 8))" \
  --minimum-messages 5 \
  >"$log_dir/probe.log" 2>&1 &
probe_process_id="$!"
process_ids+=("$probe_process_id")

ros2 bag play "$bag_path" \
  --clock 100 \
  --playback-duration "$duration_sec" \
  --disable-keyboard-controls \
  --topics /camera/color/image_raw \
  >"$log_dir/bag.log" 2>&1

wait "$probe_process_id"
cat "$log_dir/probe.log"
echo "调试日志: $log_dir"
