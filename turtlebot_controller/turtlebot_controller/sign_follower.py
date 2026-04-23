"""Drive forward and react to road signs.

Behavior
--------
Drive forward at ``linear_speed_mps`` (default 3 in/s ~ 0.0762 m/s).
When the robot gets within ``stop_distance_m`` (default 1 ft ~ 0.3048 m)
of a calibrated sign, react based on its class:

* ``Goal``         -> stop permanently and exit.
* ``Stop``         -> halt while the Stop sign is still visible; resume
                       driving after the sign has been absent for
                       ``stop_release_s``.
* ``Turn``         -> rotate 90 degrees right, then keep driving.
* ``Turn Around``  -> rotate 180 degrees, then keep driving.

Distance is computed from a YAML calibration file produced by
``calibrate_signs``::

    distance_m = reference_distance_m * ref_size_px / current_size_px

Run with::

    ros2 run turtlebot_controller sign_follower --ros-args \\
        -p calibration_file:=$HOME/sign_calibration.yaml

Parameters::

    calibration_file       string   ''       REQUIRED; YAML from calibrate_signs
    linear_speed_mps       double   0.0762   forward speed (= 3 in/s)
    stop_distance_m        double   0.3048   trigger threshold (= 1 ft)
    angular_speed_rps      double   0.6      rotation speed for turns
    min_confidence         double   0.50     drop YOLO msgs below this
    stable_frames          int      2        require N consecutive good frames
    post_turn_cooldown_s   double   1.5      ignore signs for this long after a turn
    stop_release_s         double   1.0      resume when Stop absent this long
    size_dimension         string   'width'  'width' | 'height' | 'max'
    detection_topic        string   '/yolo_detections'
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from turtlebot_controller_msgs.action import Drive, Rotate


CANON_CLASSES = {'Stop', 'Goal', 'Turn', 'Turn Around'}

# Default numbers.
DEFAULT_LINEAR_SPEED_MPS = 0.0762   # 3 in/s
DEFAULT_STOP_DISTANCE_M = 0.3048    # 1 ft
DEFAULT_ANGULAR_SPEED_RPS = 0.6
RIGHT_TURN_RAD = -math.pi / 2.0     # 90 deg right
TURN_AROUND_RAD = math.pi           # 180 deg


# States.
S_DRIVING = 'DRIVING'
S_STOPPED = 'STOPPED'
S_TURNING = 'TURNING'
S_TURNING_AROUND = 'TURNING_AROUND'
S_FINISHED = 'FINISHED'


class SignFollower(Node):
    def __init__(self) -> None:
        super().__init__('sign_follower')

        self.declare_parameter('calibration_file', '')
        self.declare_parameter('linear_speed_mps', DEFAULT_LINEAR_SPEED_MPS)
        self.declare_parameter('stop_distance_m', DEFAULT_STOP_DISTANCE_M)
        self.declare_parameter('angular_speed_rps', DEFAULT_ANGULAR_SPEED_RPS)
        self.declare_parameter('min_confidence', 0.50)
        self.declare_parameter('stable_frames', 2)
        self.declare_parameter('post_turn_cooldown_s', 1.5)
        self.declare_parameter('stop_release_s', 1.0)
        self.declare_parameter('size_dimension', 'width')
        self.declare_parameter('detection_topic', '/yolo_detections')

        p = self.get_parameter
        calib_path = p('calibration_file').get_parameter_value().string_value.strip()
        self._linear_speed = float(
            p('linear_speed_mps').get_parameter_value().double_value
        )
        self._stop_distance_m = float(
            p('stop_distance_m').get_parameter_value().double_value
        )
        self._angular_speed = float(
            p('angular_speed_rps').get_parameter_value().double_value
        )
        self._min_conf = float(
            p('min_confidence').get_parameter_value().double_value
        )
        self._stable_needed = max(
            1, int(p('stable_frames').get_parameter_value().integer_value)
        )
        self._post_turn_cooldown_s = float(
            p('post_turn_cooldown_s').get_parameter_value().double_value
        )
        self._stop_release_s = float(
            p('stop_release_s').get_parameter_value().double_value
        )
        self._size_dimension = (
            p('size_dimension').get_parameter_value().string_value or 'width'
        ).strip().lower()
        if self._size_dimension not in ('width', 'height', 'max'):
            self._size_dimension = 'width'
        detection_topic = p('detection_topic').get_parameter_value().string_value

        if not calib_path:
            self.get_logger().error(
                "'calibration_file' parameter is required. Run calibrate_signs first."
            )
            raise SystemExit(2)
        self._calibration_file = os.path.expanduser(calib_path)
        self._calib_reference_distance_m: float = 0.0
        self._calib_signs: dict[str, dict] = {}
        self._load_calibration(self._calibration_file)
        if not self._calib_signs:
            raise SystemExit(2)

        self._drive_client = ActionClient(self, Drive, 'drive')
        self._rotate_client = ActionClient(self, Rotate, 'rotate')
        self._sub = self.create_subscription(
            String, detection_topic, self._on_detection, 10
        )

        # Shared state, protected by self._lock.
        self._lock = threading.Lock()
        self._state = S_DRIVING
        self._goal_handle = None
        self._pending_next_state: str | None = None  # set when we cancel on purpose
        self._stable_count = 0
        self._pending_sign: str | None = None
        self._last_stop_seen: float = 0.0
        self._cooldown_until: float = 0.0
        self._done = threading.Event()

        # A periodic timer handles "Stop sign is gone, resume driving".
        self._timer = self.create_timer(0.2, self._on_tick)

        self.get_logger().info(
            f"sign_follower ready: speed={self._linear_speed:.3f} m/s, "
            f"trigger when bbox >= calibration size "
            f"(calibrated at {self._calib_reference_distance_m:.3f} m), "
            f"classes={sorted(self._calib_signs.keys())}."
        )

    # ----------------------------------------------------------- calibration

    def _load_calibration(self, path: str) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(
                f"Failed to load calibration_file '{path}': {exc}."
            )
            return
        ref = data.get('reference_distance_m')
        signs = data.get('signs') or {}
        if not isinstance(ref, (int, float)) or ref <= 0.0:
            self.get_logger().error(
                f"calibration_file '{path}' missing valid 'reference_distance_m'."
            )
            return
        cleaned: dict[str, dict] = {}
        for cls, entry in (signs or {}).items():
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
                f"calibration_file '{path}' had no usable entries."
            )
            return
        self._calib_reference_distance_m = float(ref)
        self._calib_signs = cleaned
        missing = CANON_CLASSES - set(cleaned.keys())
        if missing:
            self.get_logger().warn(
                f"Calibration is missing canonical class(es): {sorted(missing)}. "
                'Those signs will be ignored at runtime.'
            )

    def _ref_size_for(self, cls_name: str) -> float:
        entry = self._calib_signs.get(cls_name)
        if not entry:
            return 0.0
        if self._size_dimension == 'height':
            return float(entry.get('h_px', 0.0))
        if self._size_dimension == 'max':
            return max(float(entry.get('w_px', 0.0)), float(entry.get('h_px', 0.0)))
        return float(entry.get('w_px', 0.0))

    # ------------------------------------------------------------- detection

    def _on_detection(self, msg: String) -> None:
        det = _parse_detection(msg.data)
        if det is None:
            return
        cls_name = str(det.get('class', ''))
        conf = float(det.get('conf', 0.0))
        if conf < self._min_conf or cls_name not in CANON_CLASSES:
            return
        if cls_name not in self._calib_signs:
            return

        now = time.monotonic()
        bw = float(det.get('w', 0.0))
        bh = float(det.get('h', 0.0))
        if self._size_dimension == 'height':
            size_px = bh
        elif self._size_dimension == 'max':
            size_px = max(bw, bh)
        else:
            size_px = bw
        if size_px <= 0.0:
            return
        ref_size = self._ref_size_for(cls_name)
        if ref_size <= 0.0:
            return

        # Direct bbox comparison: bigger-than-calibration means the sign is
        # closer than the calibration distance (pinhole geometry). We still
        # compute an estimated distance for the log, but the trigger is the
        # pixel-size comparison itself.
        distance_m = self._calib_reference_distance_m * ref_size / size_px
        trigger_ratio = size_px / ref_size   # >= 1.0 means "at or past" calib

        with self._lock:
            state = self._state

            # In STOPPED state, we only care about Stop-sign presence.
            if state == S_STOPPED:
                if cls_name == 'Stop' and trigger_ratio >= 1.0:
                    self._last_stop_seen = now
                return

            if state != S_DRIVING:
                return  # ignore everything while turning or cooldown-pending

            # Cooldown: ignore detections for a bit after a turn.
            if now < self._cooldown_until:
                return

            qualifies = trigger_ratio >= 1.0
            if not qualifies:
                # Reset the stability counter if either the class changes
                # or the current one moves out of range.
                if cls_name != self._pending_sign:
                    self._pending_sign = cls_name
                self._stable_count = 0
                return

            if cls_name != self._pending_sign:
                self._pending_sign = cls_name
                self._stable_count = 1
            else:
                self._stable_count += 1

            self.get_logger().info(
                f"'{cls_name}' conf={conf:.2f} "
                f"bbox_{self._size_dimension}={size_px:.0f}px "
                f"(ref={ref_size:.0f}px, ratio={trigger_ratio:.2f}, "
                f"~{distance_m:.2f} m) "
                f"[{self._stable_count}/{self._stable_needed}]"
            )

            if self._stable_count >= self._stable_needed:
                self._trigger(cls_name, distance_m)

    def _trigger(self, cls_name: str, distance_m: float) -> None:
        """Assumes self._lock is held. Cancels Drive and schedules the
        next state based on the sign class."""
        self.get_logger().info(
            f"TRIGGER: '{cls_name}' at ~{distance_m:.2f} m."
        )
        if cls_name == 'Stop':
            self._pending_next_state = S_STOPPED
        elif cls_name == 'Goal':
            self._pending_next_state = S_FINISHED
        elif cls_name == 'Turn':
            self._pending_next_state = S_TURNING
        elif cls_name == 'Turn Around':
            self._pending_next_state = S_TURNING_AROUND
        else:
            self._pending_next_state = None
            return
        self._cancel_goal()

    # ---------------------------------------------------------------- timer

    def _on_tick(self) -> None:
        """Periodic housekeeping: Stop release + kick transitions."""
        with self._lock:
            state = self._state

        now = time.monotonic()
        if state == S_STOPPED:
            with self._lock:
                if now - self._last_stop_seen >= self._stop_release_s:
                    self.get_logger().info(
                        'Stop sign not seen for '
                        f'{now - self._last_stop_seen:.1f} s; resuming drive.'
                    )
                    self._state = S_DRIVING
                    # Apply cooldown so we don't instantly re-trigger on a
                    # lingering Stop frame after we start moving.
                    self._cooldown_until = now + self._post_turn_cooldown_s
                    self._stable_count = 0
                    self._pending_sign = None
            if self._state == S_DRIVING:
                self._send_drive_goal()

    # ----------------------------------------------------------- drive/rotate

    def _send_drive_goal(self) -> None:
        if not self._drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/drive action server not available.")
            self._finish()
            return
        goal = Drive.Goal()
        goal.distance = 0.0
        goal.speed = self._linear_speed
        self.get_logger().info(
            f"Sending Drive (distance=0, speed={self._linear_speed:.3f} m/s)."
        )
        fut = self._drive_client.send_goal_async(goal)
        fut.add_done_callback(self._on_drive_accepted)

    def _on_drive_accepted(self, future) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Drive goal rejected.')
            self._finish()
            return
        with self._lock:
            self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_drive_result)

    def _on_drive_result(self, future) -> None:
        result = future.result().result
        self.get_logger().info(
            f"Drive finished: success={result.success}, "
            f"traveled={result.distance_traveled:.3f} m, "
            f"msg='{result.message}'"
        )
        with self._lock:
            self._goal_handle = None
            next_state = self._pending_next_state
            self._pending_next_state = None
            self._stable_count = 0
            self._pending_sign = None

        if next_state == S_STOPPED:
            with self._lock:
                self._state = S_STOPPED
                self._last_stop_seen = time.monotonic()
            self.get_logger().info('Entering STOPPED; waiting for Stop sign to clear.')
        elif next_state == S_FINISHED:
            self.get_logger().info('Goal reached. Exiting.')
            self._finish()
        elif next_state == S_TURNING:
            with self._lock:
                self._state = S_TURNING
            self._send_rotate_goal(RIGHT_TURN_RAD, 'right 90 deg')
        elif next_state == S_TURNING_AROUND:
            with self._lock:
                self._state = S_TURNING_AROUND
            self._send_rotate_goal(TURN_AROUND_RAD, '180 deg')
        else:
            # Drive finished on its own (e.g. odom watchdog); exit.
            self.get_logger().warn('Drive ended without a pending transition; exiting.')
            self._finish()

    def _send_rotate_goal(self, angle_rad: float, label: str) -> None:
        if not self._rotate_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/rotate action server not available.")
            self._finish()
            return
        goal = Rotate.Goal()
        goal.angle = float(angle_rad)
        goal.angular_speed = float(self._angular_speed)
        self.get_logger().info(
            f"Sending Rotate ({label}, angle={math.degrees(angle_rad):+.1f} deg)."
        )
        fut = self._rotate_client.send_goal_async(goal)
        fut.add_done_callback(self._on_rotate_accepted)

    def _on_rotate_accepted(self, future) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Rotate goal rejected.')
            self._finish()
            return
        with self._lock:
            self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_rotate_result)

    def _on_rotate_result(self, future) -> None:
        result = future.result().result
        self.get_logger().info(
            f"Rotate finished: success={result.success}, "
            f"turned={math.degrees(result.angle_turned):+.2f} deg, "
            f"msg='{result.message}'"
        )
        with self._lock:
            self._goal_handle = None
            self._state = S_DRIVING
            self._cooldown_until = time.monotonic() + self._post_turn_cooldown_s
            self._stable_count = 0
            self._pending_sign = None
        self._send_drive_goal()

    def _cancel_goal(self) -> None:
        handle = None
        with self._lock:
            handle = self._goal_handle
        if handle is not None:
            handle.cancel_goal_async()

    def _finish(self) -> None:
        with self._lock:
            self._state = S_FINISHED
        self._done.set()


def _parse_detection(data: str) -> dict | None:
    if not data:
        return None
    data = data.strip()
    if not data.startswith('{'):
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    if 'class' in obj and 'conf' in obj:
        return obj
    return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SignFollower()

    executor_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    executor_thread.start()

    try:
        node._send_drive_goal()
        while not node._done.is_set():
            node._done.wait(timeout=0.2)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C; canceling any active goal...')
        node._cancel_goal()
        # Give cancel time to propagate before tearing down.
        time.sleep(0.5)

    # Stop rclpy first so the spin thread unwinds cleanly, THEN destroy
    # the node. Destroying while rclpy.spin is active can segfault rmw.
    if rclpy.ok():
        rclpy.shutdown()
    executor_thread.join(timeout=2.0)
    try:
        node.destroy_node()
    except Exception:
        pass

    sys.exit(0)


if __name__ == '__main__':
    main()
