"""Drive forward until the robot reaches a sign, then stop.

Uses the ``/drive`` action (Drive.action with ``distance <= 0`` drives
forever until canceled) and the ``/yolo_detections`` topic published by
``yolo_detector``. Each detection message is a JSON string like::

    {"class":"stop","class_id":0,"conf":0.87,
     "cx":320.0,"cy":240.0,"w":84.2,"h":86.0,
     "img_w":640,"img_h":480}

Three stopping modes, chosen automatically by what you pass in:

1. **Calibration-file mode** (simplest). Pass a YAML written by
   ``calibrate_signs`` via ``calibration_file``. No camera calibration
   and no physical sign measurements needed. Distance is computed
   per-class from the recorded reference bbox size at a known reference
   distance::

       distance_m = ref_distance_m * ref_size_px / current_size_px

2. **Pinhole mode**. Set ``sign_real_size_m``, ``focal_length_px``, and
   ``stop_distance_m``. All three must be > 0. Distance::

       distance_m = (sign_real_size_m * focal_length_px) / bbox_pixels

3. **Detection mode** (fallback). If neither of the above is fully
   specified, stop the first time a qualifying detection is seen for
   ``stable_frames`` consecutive frames.

Run with::

    ros2 run turtlebot_controller drive_until_sign --ros-args \\
        -p calibration_file:=$HOME/sign_calibration.yaml \\
        -p stop_distance_m:=0.30 \\
        -p linear_speed:=0.12

Parameters (all optional)::

    linear_speed       double   0.15    forward speed (m/s)
    min_confidence     double   0.50    drop YOLO msgs below this
    target_class       string   ''      only react to this class ('' = any)
    stable_frames      int      2       require N consecutive good frames
    detection_topic    string   '/yolo_detections'
    stop_distance_m    double   0.0     stop when est. distance <= this (m)
    size_dimension     string   'width' which bbox dim to use:
                                        'width', 'height', or 'max'

    # Calibration-file mode (recommended):
    calibration_file   string   ''      path to YAML from calibrate_signs

    # Pinhole mode (alternative):
    sign_real_size_m   double   0.0     real-world sign dimension (m)
    focal_length_px    double   0.0     camera focal length (pixels)
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from turtlebot_controller_msgs.action import Drive


class DriveUntilSign(Node):
    """Drive forward on /drive, stop on qualifying /yolo_detections."""

    def __init__(self) -> None:
        super().__init__('drive_until_sign')

        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('min_confidence', 0.50)
        self.declare_parameter('target_class', '')
        self.declare_parameter('stable_frames', 2)
        self.declare_parameter('detection_topic', '/yolo_detections')
        self.declare_parameter('sign_real_size_m', 0.0)
        self.declare_parameter('focal_length_px', 0.0)
        self.declare_parameter('stop_distance_m', 0.0)
        self.declare_parameter('size_dimension', 'width')
        self.declare_parameter('calibration_file', '')

        p = self.get_parameter
        self._linear_speed = float(p('linear_speed').get_parameter_value().double_value)
        self._min_conf = float(p('min_confidence').get_parameter_value().double_value)
        self._target_class = (
            p('target_class').get_parameter_value().string_value
        ).strip()
        self._stable_needed = max(
            1, int(p('stable_frames').get_parameter_value().integer_value)
        )
        detection_topic = p('detection_topic').get_parameter_value().string_value
        self._sign_real_size_m = float(
            p('sign_real_size_m').get_parameter_value().double_value
        )
        self._focal_length_px = float(
            p('focal_length_px').get_parameter_value().double_value
        )
        self._stop_distance_m = float(
            p('stop_distance_m').get_parameter_value().double_value
        )
        self._size_dimension = (
            p('size_dimension').get_parameter_value().string_value or 'width'
        ).strip().lower()
        if self._size_dimension not in ('width', 'height', 'max'):
            self.get_logger().warn(
                f"Unknown size_dimension '{self._size_dimension}', using 'width'."
            )
            self._size_dimension = 'width'

        calib_path = (
            p('calibration_file').get_parameter_value().string_value
        ).strip()
        self._calibration_file = os.path.expanduser(calib_path) if calib_path else ''
        self._calib_reference_distance_m: float = 0.0
        self._calib_signs: dict[str, dict] = {}
        if self._calibration_file:
            self._load_calibration(self._calibration_file)

        # Mode resolution: calibration file > pinhole > detection.
        if self._calib_signs and self._stop_distance_m > 0.0:
            self._mode = 'calibration'
        elif (
            self._sign_real_size_m > 0.0
            and self._focal_length_px > 0.0
            and self._stop_distance_m > 0.0
        ):
            self._mode = 'pinhole'
        else:
            self._mode = 'detection'

        self._drive_client = ActionClient(self, Drive, 'drive')
        self._sub = self.create_subscription(
            String, detection_topic, self._on_detection, 10
        )

        self._goal_handle = None
        self._done = threading.Event()
        self._stop_requested = False
        self._stable_count = 0
        self._trigger_reason = ''
        self._result_success = False

        self.get_logger().info(
            f"drive_until_sign ready [{self._mode} mode]: "
            f"speed={self._linear_speed:.2f} m/s, "
            f"min_conf={self._min_conf:.2f}, "
            f"target_class='{self._target_class or '<any>'}', "
            f"stable_frames={self._stable_needed}."
        )
        if self._mode == 'calibration':
            self.get_logger().info(
                f"calibration: file='{self._calibration_file}', "
                f"reference_distance_m={self._calib_reference_distance_m:.4f}, "
                f"stop_distance_m={self._stop_distance_m:.3f}, "
                f"size_dimension='{self._size_dimension}', "
                f"classes={list(self._calib_signs.keys())}."
            )
        elif self._mode == 'pinhole':
            self.get_logger().info(
                f"pinhole: sign_real_size_m={self._sign_real_size_m:.3f}, "
                f"focal_length_px={self._focal_length_px:.1f}, "
                f"stop_distance_m={self._stop_distance_m:.3f}, "
                f"size_dimension='{self._size_dimension}'."
            )

    # ---------------------------------------------------------------- action

    def send_goal(self) -> bool:
        if not self._drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "Drive action server '/drive' not available. "
                'Is motion_controller running?'
            )
            return False

        goal = Drive.Goal()
        goal.distance = 0.0   # 0 means "drive until canceled"
        goal.speed = self._linear_speed

        send_future = self._drive_client.send_goal_async(
            goal, feedback_callback=self._on_feedback
        )
        send_future.add_done_callback(self._on_goal_response)
        self.get_logger().info(
            f'Drive goal sent (distance=0, speed={self._linear_speed:.2f} m/s).'
        )
        return True

    def _on_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        self.get_logger().info(
            f'drive: {fb.distance_traveled:.3f} m traveled @ '
            f'{fb.current_speed:.2f} m/s'
        )

    def _on_goal_response(self, future) -> None:
        self._goal_handle = future.result()
        if self._goal_handle is None or not self._goal_handle.accepted:
            self.get_logger().error(
                'Drive goal rejected. Another Drive or Rotate goal is probably active.'
            )
            self._done.set()
            return
        self.get_logger().info('Drive goal accepted. Watching /yolo_detections...')

        if self._stop_requested:
            self._cancel_goal()

        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        result = future.result().result
        self._result_success = bool(result.success)
        self.get_logger().info(
            f"Drive finished: success={result.success}, "
            f"traveled={result.distance_traveled:.3f} m, "
            f"reason='{self._trigger_reason or result.message}'"
        )
        self._done.set()

    def _cancel_goal(self) -> None:
        if self._goal_handle is None:
            return
        self.get_logger().info(f'Canceling Drive goal ({self._trigger_reason}).')
        self._goal_handle.cancel_goal_async()

    # ---------------------------------------------------------- calibration

    def _load_calibration(self, path: str) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(
                f"Failed to load calibration_file '{path}': {exc}. "
                'Continuing without calibration.'
            )
            return

        ref = data.get('reference_distance_m')
        signs = data.get('signs') or {}
        if not isinstance(ref, (int, float)) or ref <= 0.0:
            self.get_logger().error(
                f"calibration_file '{path}' has missing or invalid "
                "'reference_distance_m'; ignoring."
            )
            return
        if not isinstance(signs, dict) or not signs:
            self.get_logger().error(
                f"calibration_file '{path}' has no 'signs' entries; ignoring."
            )
            return

        cleaned: dict[str, dict] = {}
        for cls, entry in signs.items():
            if not isinstance(entry, dict):
                continue
            try:
                w_px = float(entry.get('w_px', 0.0))
                h_px = float(entry.get('h_px', 0.0))
            except (TypeError, ValueError):
                continue
            if w_px <= 0.0 or h_px <= 0.0:
                continue
            cleaned[str(cls)] = {'w_px': w_px, 'h_px': h_px}

        if not cleaned:
            self.get_logger().error(
                f"calibration_file '{path}' had no usable entries; ignoring."
            )
            return

        self._calib_reference_distance_m = float(ref)
        self._calib_signs = cleaned

    def _reference_size_for(self, cls_name: str | None) -> float:
        """Return the reference bbox size in px for this class, or 0 if none."""
        if cls_name is None or cls_name not in self._calib_signs:
            return 0.0
        entry = self._calib_signs[cls_name]
        if self._size_dimension == 'height':
            return float(entry.get('h_px', 0.0))
        if self._size_dimension == 'max':
            return max(
                float(entry.get('w_px', 0.0)), float(entry.get('h_px', 0.0))
            )
        return float(entry.get('w_px', 0.0))

    # ------------------------------------------------------------- detection

    def _on_detection(self, msg: String) -> None:
        if self._stop_requested:
            return

        det = _parse_detection(msg.data)
        if det is None:
            return

        cls_name = det.get('class')
        conf = float(det.get('conf', 0.0))

        if conf < self._min_conf:
            self._stable_count = 0
            return
        if self._target_class and cls_name != self._target_class:
            self._stable_count = 0
            return

        bw = float(det.get('w', 0.0))
        bh = float(det.get('h', 0.0))
        if self._size_dimension == 'height':
            size_px = bh
        elif self._size_dimension == 'max':
            size_px = max(bw, bh)
        else:
            size_px = bw

        # Compute estimated distance from whichever mode is active.
        distance_m = math.inf
        if self._mode == 'calibration':
            ref_size = self._reference_size_for(cls_name)
            if ref_size > 0.0 and size_px > 0.0:
                distance_m = (
                    self._calib_reference_distance_m * ref_size / size_px
                )
        elif self._mode == 'pinhole':
            if size_px > 0.0:
                distance_m = (
                    self._sign_real_size_m * self._focal_length_px / size_px
                )

        if self._mode in ('calibration', 'pinhole'):
            if not math.isfinite(distance_m):
                self.get_logger().warn(
                    f"Got '{cls_name}' but no reference in calibration; ignoring."
                    if self._mode == 'calibration'
                    else f"Got '{cls_name}' with zero bbox; ignoring."
                )
                return
            qualifies = distance_m <= self._stop_distance_m
            self._stable_count = self._stable_count + 1 if qualifies else 0
            self.get_logger().info(
                f"'{cls_name}' conf={conf:.2f} "
                f"bbox_{self._size_dimension}={size_px:.0f}px "
                f"dist={distance_m:.2f} m "
                f"[{self._stable_count}/{self._stable_needed}]"
            )
            if qualifies and self._stable_count >= self._stable_needed:
                self._stop_requested = True
                self._trigger_reason = (
                    f"'{cls_name}' at ~{distance_m:.2f} m <= "
                    f'{self._stop_distance_m:.2f} m'
                )
                self._cancel_goal()
        else:
            self._stable_count += 1
            self.get_logger().info(
                f"'{cls_name}' conf={conf:.2f} "
                f"bbox_{self._size_dimension}={size_px:.0f}px "
                f"[{self._stable_count}/{self._stable_needed}]"
            )
            if self._stable_count >= self._stable_needed:
                self._stop_requested = True
                self._trigger_reason = (
                    f"detected '{cls_name}' with conf={conf:.2f}"
                )
                self._cancel_goal()


def _parse_detection(data: str) -> dict | None:
    """Parse a /yolo_detections payload.

    Supports the current JSON format emitted by yolo_detector.py and
    falls back to the old ``"class:conf"`` format so older bag files
    still play back. Returns a dict with at least ``class`` and
    ``conf`` keys, or ``None`` if parsing fails.
    """
    if not data:
        return None
    data = data.strip()

    # JSON format (new).
    if data.startswith('{'):
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        if 'class' in obj and 'conf' in obj:
            return obj
        return None

    # Legacy "class:conf" format.
    sep = data.rfind(':')
    if sep <= 0:
        return None
    cls = data[:sep].strip()
    try:
        conf = float(data[sep + 1:].strip())
    except ValueError:
        return None
    return {'class': cls, 'conf': conf, 'w': 0.0, 'h': 0.0}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DriveUntilSign()

    executor_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    executor_thread.start()

    try:
        if not node.send_goal():
            return
        while not node._done.is_set():
            node._done.wait(timeout=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C received; canceling Drive goal...')
        node._trigger_reason = 'Ctrl-C'
        node._cancel_goal()
        node._done.wait(timeout=5.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not node._result_success and not node._stop_requested:
        sys.exit(1)


if __name__ == '__main__':
    main()
