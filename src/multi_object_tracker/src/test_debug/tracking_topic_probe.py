#!/usr/bin/env python3

import argparse
from collections import Counter
import json
import time

from perception_interfaces.msg import Detection2DArray
from perception_interfaces.msg import TrackedObject2DArray
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


def stamp_key(header):
    return header.stamp.sec, header.stamp.nanosec


class TrackingTopicProbe(Node):
    def __init__(self):
        super().__init__('tracking_topic_probe')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.detection_messages = 0
        self.tracking_messages = 0
        self.nonempty_detection_messages = 0
        self.nonempty_tracking_messages = 0
        self.detection_stamps = set()
        self.tracking_stamps = set()
        self.id_observations = Counter()
        self.duplicate_id_messages = 0
        self.create_subscription(
            Detection2DArray,
            '/yolo/detections_2d',
            self.on_detections,
            qos,
        )
        self.create_subscription(
            TrackedObject2DArray,
            '/tracking/tracks_2d',
            self.on_tracks,
            qos,
        )

    def on_detections(self, message):
        self.detection_messages += 1
        self.nonempty_detection_messages += bool(message.detections)
        self.detection_stamps.add(stamp_key(message.header))

    def on_tracks(self, message):
        self.tracking_messages += 1
        self.nonempty_tracking_messages += bool(message.tracks)
        self.tracking_stamps.add(stamp_key(message.header))
        track_ids = [tracked_object.track_id for tracked_object in message.tracks]
        if len(track_ids) != len(set(track_ids)):
            self.duplicate_id_messages += 1
        self.id_observations.update(track_ids)

    def report(self):
        unmatched_tracking_stamps = self.tracking_stamps - self.detection_stamps
        return {
            'detection_messages': self.detection_messages,
            'tracking_messages': self.tracking_messages,
            'nonempty_detection_messages': self.nonempty_detection_messages,
            'nonempty_tracking_messages': self.nonempty_tracking_messages,
            'unique_track_ids': len(self.id_observations),
            'maximum_observations_per_id': max(self.id_observations.values(), default=0),
            'duplicate_id_messages': self.duplicate_id_messages,
            'tracking_stamps_without_detection': len(unmatched_tracking_stamps),
        }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-sec', type=float, default=20.0)
    parser.add_argument('--minimum-messages', type=int, default=5)
    parser.add_argument('--minimum-id-observations', type=int, default=3)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    rclpy.init()
    node = TrackingTopicProbe()
    deadline = time.monotonic() + arguments.timeout_sec
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    report = node.report()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    node.destroy_node()
    rclpy.shutdown()

    passed = (
        report['detection_messages'] >= arguments.minimum_messages
        and report['tracking_messages'] >= arguments.minimum_messages
        and report['nonempty_tracking_messages'] > 0
        and report['maximum_observations_per_id'] >= arguments.minimum_id_observations
        and report['duplicate_id_messages'] == 0
        and report['tracking_stamps_without_detection'] == 0
    )
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
