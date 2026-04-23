"""Capture a per-class bbox reference at a known distance (default 1 ft).

Rather than measuring each sign's physical size, just place each sign at a
fixed reference distance (1 foot by default) and let this tool record the
average bounding-box size YOLO sees. The resulting YAML can be fed into
``drive_until_sign`` via the ``calibration_file`` parameter, and distance
is then computed as::

    distance_m = reference_distance_m * ref_bbox_size_px / current_bbox_size_px

without ever knowing the physical sign dimensions.

Run with::

    ros2 run turtlebot_controller calibrate_signs --ros-args \\
        -p classes:="['stop','yield','left','right']" \\
        -p output_file:=$HOME/sign_calibration.yaml

Parameters::

    classes              string[]  classes to calibrate, in order
    reference_distance_m double    0.3048      (= 1 ft)
    output_file          string    ~/sign_calibration.yaml
    sample_seconds       double    3.0         collect window per class
    detection_topic      string    /yolo_detections
    min_confidence       double    0.30        drop samples below this
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String


FOOT_IN_METERS = 0.3048


class CalibrationCollector(Node):
    """Listens to /yolo_detections and collects samples on demand."""

    def __init__(self) -> None:
        super().__init__('calibrate_signs')

        self.declare_parameter('classes', [''])
        self.declare_parameter('reference_distance_m', FOOT_IN_METERS)
        self.declare_parameter(
            'output_file', os.path.expanduser('~/sign_calibration.yaml')
        )
        self.declare_parameter('sample_seconds', 3.0)
        self.declare_parameter('detection_topic', '/yolo_detections')
        self.declare_parameter('min_confidence', 0.30)

        p = self.get_parameter
        self.classes: list[str] = [
            c for c in p('classes').get_parameter_value().string_array_value
            if c.strip()
        ]
        self.reference_distance_m = float(
            p('reference_distance_m').get_parameter_value().double_value
        )
        self.output_file = os.path.expanduser(
            p('output_file').get_parameter_value().string_value
        )
        self.sample_seconds = float(
            p('sample_seconds').get_parameter_value().double_value
        )
        detection_topic = p('detection_topic').get_parameter_value().string_value
        self.min_confidence = float(
            p('min_confidence').get_parameter_value().double_value
        )

        self._lock = threading.Lock()
        self._collecting_class: str | None = None
        self._samples: list[tuple[float, float]] = []  # (w_px, h_px)

        self._sub = self.create_subscription(
            String, detection_topic, self._on_detection, 10
        )

        if not self.classes:
            self.get_logger().error(
                "No classes provided. Pass them with "
                "-p classes:=\"['stop','yield',...]\"."
            )
            raise SystemExit(2)

        self.get_logger().info(
            f"Calibrating {len(self.classes)} class(es) at "
            f"reference_distance_m={self.reference_distance_m:.4f} "
            f"(={self.reference_distance_m/FOOT_IN_METERS:.2f} ft). "
            f"Output: {self.output_file}"
        )

    def _on_detection(self, msg: String) -> None:
        with self._lock:
            if self._collecting_class is None:
                return
            target = self._collecting_class

        data = msg.data.strip()
        if not data.startswith('{'):
            # We only understand the JSON format (new yolo_detector output).
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return

        if obj.get('class') != target:
            return
        conf = float(obj.get('conf', 0.0))
        if conf < self.min_confidence:
            return
        w = float(obj.get('w', 0.0))
        h = float(obj.get('h', 0.0))
        if w <= 0.0 or h <= 0.0:
            return

        with self._lock:
            self._samples.append((w, h))

    def collect_for(self, cls: str) -> tuple[float, float, int]:
        """Return (mean_w_px, mean_h_px, n_samples) for a class."""
        with self._lock:
            self._collecting_class = cls
            self._samples = []

        deadline = time.monotonic() + self.sample_seconds
        while time.monotonic() < deadline:
            time.sleep(0.05)

        with self._lock:
            self._collecting_class = None
            samples = list(self._samples)
            self._samples = []

        if not samples:
            return 0.0, 0.0, 0
        ws = [s[0] for s in samples]
        hs = [s[1] for s in samples]
        return sum(ws) / len(ws), sum(hs) / len(hs), len(samples)


def _run_wizard(node: CalibrationCollector) -> dict:
    """Interactive loop. Returns the calibration dict ready to serialize."""
    result: dict = {
        'reference_distance_m': node.reference_distance_m,
        'signs': {},
    }

    ft = node.reference_distance_m / FOOT_IN_METERS
    for cls in node.classes:
        while True:
            try:
                input(
                    f"\n>>> Place class '{cls}' exactly "
                    f"{node.reference_distance_m:.4f} m "
                    f"(~{ft:.2f} ft) in front of the camera. "
                    f"Press Enter to sample (or type 's' to skip, 'r' to retry)..."
                )
            except EOFError:
                print('\nEOF on stdin; aborting.', file=sys.stderr)
                return result

            node.get_logger().info(
                f"Sampling '{cls}' for {node.sample_seconds:.1f} s..."
            )
            mean_w, mean_h, n = node.collect_for(cls)

            if n == 0:
                node.get_logger().warn(
                    f"No qualifying detections of '{cls}'. "
                    'Check the class name, confidence threshold, and that '
                    'yolo_detector is running. (r)etry, (s)kip?'
                )
                choice = input('[r/s]: ').strip().lower()
                if choice.startswith('s'):
                    break
                else:
                    continue

            node.get_logger().info(
                f"'{cls}': n={n}, mean_bbox = {mean_w:.1f} x {mean_h:.1f} px."
            )
            result['signs'][cls] = {
                'w_px': round(mean_w, 2),
                'h_px': round(mean_h, 2),
                'samples': int(n),
            }
            break

    return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CalibrationCollector()

    executor_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    executor_thread.start()

    try:
        result = _run_wizard(node)

        output_path = os.path.expanduser(node.output_file)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(result, f, sort_keys=False)

        node.get_logger().info(f"Wrote calibration to '{output_path}'.")
        print(f"\nCalibration written to {output_path}\n")
        print(yaml.safe_dump(result, sort_keys=False))
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C; exiting without writing.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
