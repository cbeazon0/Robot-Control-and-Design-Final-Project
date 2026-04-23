"""Drive forward and react to road signs.

Behavior
--------
Drive forward at ``linear_speed_mps`` (default 1.5 in/s ~ 0.0381 m/s).
As soon as the YOLO detector reports the same sign class ``stable_frames``
times in a row (default 3) above ``min_confidence``, act on it:

* ``Goal``         -> stop permanently and exit.
* ``Stop``         -> run the "go around" maneuver: wait 3 s, right 90,
                       forward 1 ft, left 90, forward 1.5 ft, left 90,
                       forward 1 ft, right 90, then resume forward drive.
* ``Turn``         -> rotate 90 degrees right, then keep driving.
* ``Turn Around``  -> rotate 180 degrees, then keep driving.

No calibration or distance estimation is used. Place signs so they only
come into YOLO's view at the distance where you want the robot to react.

Run with::

    ros2 run turtlebot_controller sign_follower

Parameters::

    linear_speed_mps       double   0.0381   forward speed (= 1.5 in/s)
    maneuver_speed_mps     double   0.08     forward speed inside the Stop maneuver
    angular_speed_rps      double   0.6      rotation speed for turns
    min_confidence         double   0.25     drop YOLO msgs below this (match yolo_detector)
    stable_frames          int      3        N consecutive same-class frames
    post_turn_cooldown_s   double   3.0      ignore signs for this long after any motion
    detection_topic        string   '/yolo_detections'
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from turtlebot_controller_msgs.action import Drive, Rotate


CANON_CLASSES = {'Stop', 'Goal', 'Turn', 'Turn Around'}

# Default numbers.
DEFAULT_LINEAR_SPEED_MPS = 0.0381   # 1.5 in/s (slowed for YOLO latency)
DEFAULT_MANEUVER_SPEED_MPS = 0.08   # a bit faster for the scripted Stop maneuver
DEFAULT_ANGULAR_SPEED_RPS = 0.6
RIGHT_TURN_RAD = -math.pi / 2.0     # 90 deg right
LEFT_TURN_RAD = math.pi / 2.0       # 90 deg left
TURN_AROUND_RAD = math.pi           # 180 deg

ONE_FT_M = 0.3048
ONE_AND_HALF_FT_M = 0.4572


# States.
S_DRIVING = 'DRIVING'
S_TURNING = 'TURNING'
S_TURNING_AROUND = 'TURNING_AROUND'
S_MANEUVER = 'MANEUVER'
S_FINISHED = 'FINISHED'


# Stop maneuver: list of (kind, *args) tuples.
#   ('wait',   seconds)
#   ('rotate', angle_rad, label)
#   ('drive',  distance_m, label)
def _stop_maneuver_steps() -> list[tuple]:
    return [
        ('wait',   3.0),
        ('rotate', RIGHT_TURN_RAD, 'stop-maneuver: right 90'),
        ('drive',  ONE_FT_M, 'stop-maneuver: forward 1 ft'),
        ('rotate', LEFT_TURN_RAD, 'stop-maneuver: left 90'),
        ('drive',  ONE_AND_HALF_FT_M, 'stop-maneuver: forward 1.5 ft'),
        ('rotate', LEFT_TURN_RAD, 'stop-maneuver: left 90'),
        ('drive',  ONE_FT_M, 'stop-maneuver: forward 1 ft'),
        ('rotate', RIGHT_TURN_RAD, 'stop-maneuver: right 90'),
    ]


class SignFollower(Node):
    def __init__(self) -> None:
        super().__init__('sign_follower')

        self.declare_parameter('linear_speed_mps', DEFAULT_LINEAR_SPEED_MPS)
        self.declare_parameter('maneuver_speed_mps', DEFAULT_MANEUVER_SPEED_MPS)
        self.declare_parameter('angular_speed_rps', DEFAULT_ANGULAR_SPEED_RPS)
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('post_turn_cooldown_s', 3.0)
        self.declare_parameter('detection_topic', '/yolo_detections')

        p = self.get_parameter
        self._linear_speed = float(
            p('linear_speed_mps').get_parameter_value().double_value
        )
        self._maneuver_speed = float(
            p('maneuver_speed_mps').get_parameter_value().double_value
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
        detection_topic = p('detection_topic').get_parameter_value().string_value

        self._drive_client = ActionClient(self, Drive, 'drive')
        self._rotate_client = ActionClient(self, Rotate, 'rotate')
        self._sub = self.create_subscription(
            String, detection_topic, self._on_detection, 10
        )

        # Shared state, protected by self._lock. RLock because some paths
        # (detection -> _trigger -> _cancel_goal) re-enter the lock.
        self._lock = threading.RLock()
        self._state = S_DRIVING
        self._goal_handle = None
        self._pending_next_state: str | None = None  # set when we cancel on purpose
        self._stable_count = 0
        self._pending_sign: str | None = None
        self._cooldown_until: float = 0.0
        self._maneuver_steps: list[tuple] = []
        self._wait_timer = None
        self._done = threading.Event()

        self.get_logger().info(
            f"sign_follower ready: speed={self._linear_speed:.3f} m/s, "
            f"trigger after {self._stable_needed} consecutive detections "
            f"(conf >= {self._min_conf:.2f}) of the same sign."
        )

    # ------------------------------------------------------------- detection

    def _on_detection(self, msg: String) -> None:
        det = _parse_detection(msg.data)
        if det is None:
            return
        cls_name = str(det.get('class', ''))
        conf = float(det.get('conf', 0.0))
        if conf < self._min_conf:
            self.get_logger().info(
                f"drop '{cls_name}' conf={conf:.2f} "
                f"(below min_confidence={self._min_conf:.2f})"
            )
            return
        if cls_name not in CANON_CLASSES:
            self.get_logger().info(
                f"drop class '{cls_name}' (not in {sorted(CANON_CLASSES)})"
            )
            return

        now = time.monotonic()

        with self._lock:
            if self._state != S_DRIVING:
                return  # ignore detections while turning / mid-maneuver

            # Cooldown: ignore detections for a bit after a turn/maneuver.
            if now < self._cooldown_until:
                return

            if cls_name != self._pending_sign:
                self._pending_sign = cls_name
                self._stable_count = 1
            else:
                self._stable_count += 1

            self.get_logger().info(
                f"'{cls_name}' conf={conf:.2f} "
                f"[{self._stable_count}/{self._stable_needed}]"
            )

            if self._stable_count >= self._stable_needed:
                self._trigger(cls_name)

    def _trigger(self, cls_name: str) -> None:
        """Assumes self._lock is held. Cancels Drive and schedules the
        next state based on the sign class."""
        self.get_logger().info(f"TRIGGER: '{cls_name}'.")
        if cls_name == 'Stop':
            self._maneuver_steps = _stop_maneuver_steps()
            self._pending_next_state = S_MANEUVER
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

    # ----------------------------------------------------------- maneuver

    def _run_next_step(self) -> None:
        """Pop and execute the next step of the Stop maneuver."""
        with self._lock:
            if not self._maneuver_steps:
                self._state = S_DRIVING
                self._cooldown_until = (
                    time.monotonic() + self._post_turn_cooldown_s
                )
                self._stable_count = 0
                self._pending_sign = None
                self.get_logger().info(
                    'Stop maneuver complete; resuming forward drive.'
                )
                resume = True
                step = None
            else:
                step = self._maneuver_steps.pop(0)
                resume = False

        if resume:
            self._send_drive_goal()
            return

        kind = step[0]
        if kind == 'wait':
            secs = float(step[1])
            self.get_logger().info(f'Maneuver: wait {secs:.1f} s')
            self._start_wait(secs)
        elif kind == 'rotate':
            angle = float(step[1])
            label = str(step[2])
            self._send_rotate_goal(angle, label)
        elif kind == 'drive':
            distance = float(step[1])
            label = str(step[2])
            self._send_drive_distance_goal(distance, label)
        else:
            self.get_logger().error(f'Unknown maneuver step: {step!r}')
            self._finish()

    def _start_wait(self, seconds: float) -> None:
        # One-shot timer; we cancel + destroy it in the callback.
        def _after() -> None:
            t = self._wait_timer
            self._wait_timer = None
            if t is not None:
                try:
                    t.cancel()
                    self.destroy_timer(t)
                except Exception:
                    pass
            self._run_next_step()

        self._wait_timer = self.create_timer(max(0.0, seconds), _after)

    # ----------------------------------------------------------- drive/rotate

    def _send_drive_goal(self) -> None:
        """Send an open-ended Drive (distance=0) at the cruising speed."""
        self._send_drive(
            distance=0.0, speed=self._linear_speed, label='cruise'
        )

    def _send_drive_distance_goal(self, distance_m: float, label: str) -> None:
        """Send a finite Drive at the maneuver speed."""
        self._send_drive(
            distance=float(distance_m),
            speed=self._maneuver_speed,
            label=label,
        )

    def _send_drive(self, distance: float, speed: float, label: str) -> None:
        if not self._drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/drive action server not available.")
            self._finish()
            return
        goal = Drive.Goal()
        goal.distance = float(distance)
        goal.speed = float(speed)
        self.get_logger().info(
            f"Sending Drive ({label}: distance={distance:.3f} m, "
            f"speed={speed:.3f} m/s)."
        )
        fut = self._drive_client.send_goal_async(goal)
        fut.add_done_callback(self._on_drive_accepted)

    def _on_drive_accepted(self, future) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(
                'Drive goal rejected. Check: (1) is motion_controller running? '
                "(2) 'ros2 action info /drive' shows exactly one server?"
            )
            self._finish()
            return
        pending_cancel = False
        with self._lock:
            self._goal_handle = handle
            # If we tried to cancel before accept completed, do it now.
            pending_cancel = self._pending_next_state is not None
        self.get_logger().info('Drive goal accepted.')
        handle.get_result_async().add_done_callback(self._on_drive_result)
        if pending_cancel:
            self.get_logger().info('(Applying deferred cancel on Drive.)')
            handle.cancel_goal_async()

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
            state = self._state

        if next_state == S_FINISHED:
            self.get_logger().info('Goal reached. Exiting.')
            self._finish()
            return
        if next_state == S_TURNING:
            with self._lock:
                self._state = S_TURNING
                self._stable_count = 0
                self._pending_sign = None
            self._send_rotate_goal(RIGHT_TURN_RAD, 'right 90 deg')
            return
        if next_state == S_TURNING_AROUND:
            with self._lock:
                self._state = S_TURNING_AROUND
                self._stable_count = 0
                self._pending_sign = None
            self._send_rotate_goal(TURN_AROUND_RAD, '180 deg')
            return
        if next_state == S_MANEUVER:
            with self._lock:
                self._state = S_MANEUVER
                self._stable_count = 0
                self._pending_sign = None
            self._run_next_step()
            return

        if state == S_MANEUVER:
            # A maneuver sub-step drive finished; continue the sequence.
            self._run_next_step()
            return

        # Drive ended on its own (e.g. odom watchdog) while cruising; exit.
        self.get_logger().warn(
            'Drive ended without a pending transition; exiting.'
        )
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
            state = self._state

        if state == S_MANEUVER:
            self._run_next_step()
            return

        # Normal Turn / Turn Around: resume cruising.
        with self._lock:
            self._state = S_DRIVING
            self._cooldown_until = (
                time.monotonic() + self._post_turn_cooldown_s
            )
            self._stable_count = 0
            self._pending_sign = None
        self._send_drive_goal()

    def _cancel_goal(self) -> None:
        handle = None
        with self._lock:
            handle = self._goal_handle
        if handle is not None:
            self.get_logger().info('Requesting goal cancel.')
            handle.cancel_goal_async()
        else:
            # Accept callback hasn't fired yet. _on_drive_accepted will
            # see _pending_next_state and cancel once the handle exists.
            self.get_logger().info(
                'Cancel requested before goal handle was available; '
                'will cancel on accept.'
            )

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
