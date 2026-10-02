#!/usr/bin/env python3

import argparse
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Publish one image repeatedly for detector debugging.'
    )
    parser.add_argument('image', type=Path)
    parser.add_argument('--topic', default='/test/image')
    parser.add_argument('--count', type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f'Could not read image: {args.image}')

    rclpy.init()
    node = rclpy.create_node('yolo_test_image_publisher')
    publisher = node.create_publisher(Image, args.topic, qos_profile_sensor_data)
    bridge = CvBridge()

    deadline = time.monotonic() + 5.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if publisher.get_subscription_count() == 0:
        raise RuntimeError(f'No subscriber discovered on {args.topic}')
    time.sleep(1.0)

    for _ in range(args.count):
        message = bridge.cv2_to_imgmsg(image, encoding='bgr8')
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = 'test_camera'
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.2)
        time.sleep(0.2)

    time.sleep(1.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
