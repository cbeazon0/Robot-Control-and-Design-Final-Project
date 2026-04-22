"""Closed-loop motion controller with Drive and Rotate action servers.

Exposes two ROS2 action servers:

* ``/drive``  (``turtlebot_controller_msgs/action/Drive``)
    Drives forward a fixed distance using odometry, or indefinitely until
    canceled if ``distance <= 0``.

* ``/rotate`` (``turtlebot_controller_msgs/action/Rotate``)
    Rotates in place by a signed angle (radians).

Both use a simple P-controller that naturally tapers as the goal is
approached, and both publish ``geometry_msgs/Twist`` on ``/cmd_vel``.

Only one goal (Drive or Rotate) runs at a time; concurrent goals are
rejected so the two action servers cannot fight over ``/cmd_vel``.

A watchdog aborts the active goal and stops the robot if odometry stops
arriving, and the node publishes a zero ``Twist`` on shutdown.
"""

from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from turtlebot_controller_msgs.action import Drive, Rotate


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return yaw (rotation about +Z) in radians from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def shortest_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi] using a numerically stable form."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MotionController(Node):
    def __init__(self) -> None:
        super().__init__('motion_controller')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('max_linear_speed', 0.15)
        self.declare_parameter('max_angular_speed', 0.6)
        self.declare_parameter('linear_kp', 0.8)
        self.declare_parameter('angular_kp', 1.5)
        self.declare_parameter('distance_tolerance', 0.02)
        self.declare_parameter('angle_tolerance', 0.02)
        self.declare_parameter('odom_timeout_s', 0.5)

        self._odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self._cmd_vel_topic = (
            self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        )
        self._control_rate = self.get_parameter('control_rate_hz').get_parameter_value().double_value
        self._max_linear = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self._max_angular = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self._linear_kp = self.get_parameter('linear_kp').get_parameter_value().double_value
        self._angular_kp = self.get_parameter('angular_kp').get_parameter_value().double_value
        self._dist_tol = self.get_parameter('distance_tolerance').get_parameter_value().double_value
        self._angle_tol = self.get_parameter('angle_tolerance').get_parameter_value().double_value
        self._odom_timeout = self.get_parameter('odom_timeout_s').get_parameter_value().double_value

        self._cb_group = ReentrantCallbackGroup()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        self._odom_sub = self.create_subscription(
            Odometry,
            self._odom_topic,
            self._on_odom,
            sensor_qos,
            callback_group=self._cb_group,
        )

        self._state_lock = threading.Lock()
        self._have_odom = False
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_odom_monotonic: float = 0.0

        self._goal_lock = threading.Lock()
        self._goal_active = False

        self._drive_server = ActionServer(
            self,
            Drive,
            'drive',
            execute_callback=self._execute_drive,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self._cb_group,
        )

        self._rotate_server = ActionServer(
            self,
            Rotate,
            'rotate',
            execute_callback=self._execute_rotate,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"motion_controller ready. odom='{self._odom_topic}', cmd_vel='{self._cmd_vel_topic}', "
            f"rate={self._control_rate:.1f} Hz, max_lin={self._max_linear:.2f} m/s, "
            f"max_ang={self._max_angular:.2f} rad/s."
        )

    # ------------------------------------------------------------------ odom

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        with self._state_lock:
            self._x = msg.pose.pose.position.x
            self._y = msg.pose.pose.position.y
            self._yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            self._have_odom = True
            self._last_odom_monotonic = time.monotonic()

    def _snapshot(self) -> tuple[bool, float, float, float, float]:
        """Return (have_odom, x, y, yaw, age_s)."""
        with self._state_lock:
            age = time.monotonic() - self._last_odom_monotonic if self._have_odom else float('inf')
            return self._have_odom, self._x, self._y, self._yaw, age

    # ---------------------------------------------------------------- helpers

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _accept_goal(self, _goal_request) -> GoalResponse:
        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warn(
                    'Rejecting new goal: another Drive/Rotate goal is already active.'
                )
                return GoalResponse.REJECT
            self._goal_active = True
            return GoalResponse.ACCEPT

    def _accept_cancel(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _release_goal(self) -> None:
        with self._goal_lock:
            self._goal_active = False

    def _wait_for_odom(self, timeout_s: float = 5.0) -> bool:
        """Block until at least one odom message has been received (or timeout)."""
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            with self._state_lock:
                if self._have_odom:
                    return True
            time.sleep(0.02)
        return False

    # ----------------------------------------------------------------- drive

    def _execute_drive(self, goal_handle):
        goal = goal_handle.request
        requested_speed = abs(float(goal.speed)) if goal.speed else self._max_linear
        speed_cap = clamp(requested_speed, 0.0, self._max_linear)
        target_distance = float(goal.distance)
        indefinite = target_distance <= 0.0

        result = Drive.Result()

        try:
            if speed_cap <= 0.0:
                goal_handle.abort()
                result.success = False
                result.message = 'speed must be > 0'
                result.distance_traveled = 0.0
                return result

            if not self._wait_for_odom():
                goal_handle.abort()
                self._publish_stop()
                result.success = False
                result.message = 'No odometry received; cannot run closed-loop drive.'
                result.distance_traveled = 0.0
                return result

            have, x0, y0, _yaw0, _age = self._snapshot()
            if not have:
                goal_handle.abort()
                self._publish_stop()
                result.success = False
                result.message = 'No odometry snapshot.'
                result.distance_traveled = 0.0
                return result

            mode = 'indefinite' if indefinite else f'{target_distance:.3f} m'
            self.get_logger().info(
                f'Drive goal: {mode} at up to {speed_cap:.3f} m/s.'
            )

            period = 1.0 / max(self._control_rate, 1.0)
            traveled = 0.0

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self._publish_stop()
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Canceled by client.'
                    result.distance_traveled = traveled
                    self.get_logger().info(f'Drive canceled after {traveled:.3f} m.')
                    return result

                have, x, y, _yaw, age = self._snapshot()
                if not have or age > self._odom_timeout:
                    self._publish_stop()
                    goal_handle.abort()
                    result.success = False
                    result.message = f'Odometry watchdog triggered (age={age:.2f}s).'
                    result.distance_traveled = traveled
                    self.get_logger().error(result.message)
                    return result

                traveled = math.hypot(x - x0, y - y0)

                if indefinite:
                    v = speed_cap
                    remaining = float('inf')
                else:
                    remaining = target_distance - traveled
                    if remaining <= self._dist_tol:
                        self._publish_stop()
                        goal_handle.succeed()
                        result.success = True
                        result.message = 'Reached target distance.'
                        result.distance_traveled = traveled
                        self.get_logger().info(
                            f'Drive succeeded: traveled {traveled:.3f} m (target {target_distance:.3f} m).'
                        )
                        return result
                    v = clamp(self._linear_kp * remaining, 0.0, speed_cap)

                cmd = Twist()
                cmd.linear.x = v
                self._cmd_pub.publish(cmd)

                fb = Drive.Feedback()
                fb.distance_traveled = traveled
                fb.distance_remaining = -1.0 if indefinite else max(0.0, remaining)
                fb.current_speed = v
                goal_handle.publish_feedback(fb)

                time.sleep(period)

            self._publish_stop()
            goal_handle.abort()
            result.success = False
            result.message = 'Shutting down.'
            result.distance_traveled = traveled
            return result
        finally:
            self._publish_stop()
            self._release_goal()

    # ---------------------------------------------------------------- rotate

    def _execute_rotate(self, goal_handle):
        goal = goal_handle.request
        target_angle = float(goal.angle)
        requested_speed = abs(float(goal.angular_speed)) if goal.angular_speed else self._max_angular
        speed_cap = clamp(requested_speed, 0.0, self._max_angular)

        result = Rotate.Result()

        try:
            if speed_cap <= 0.0:
                goal_handle.abort()
                result.success = False
                result.message = 'angular_speed must be > 0'
                result.angle_turned = 0.0
                return result

            if not self._wait_for_odom():
                goal_handle.abort()
                self._publish_stop()
                result.success = False
                result.message = 'No odometry received; cannot run closed-loop rotate.'
                result.angle_turned = 0.0
                return result

            have, _x, _y, yaw0, _age = self._snapshot()
            if not have:
                goal_handle.abort()
                self._publish_stop()
                result.success = False
                result.message = 'No odometry snapshot.'
                result.angle_turned = 0.0
                return result

            self.get_logger().info(
                f'Rotate goal: {math.degrees(target_angle):+.1f} deg at up to '
                f'{speed_cap:.2f} rad/s.'
            )

            period = 1.0 / max(self._control_rate, 1.0)
            turned = 0.0

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self._publish_stop()
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Canceled by client.'
                    result.angle_turned = turned
                    self.get_logger().info(
                        f'Rotate canceled after {math.degrees(turned):+.1f} deg.'
                    )
                    return result

                have, _x, _y, yaw, age = self._snapshot()
                if not have or age > self._odom_timeout:
                    self._publish_stop()
                    goal_handle.abort()
                    result.success = False
                    result.message = f'Odometry watchdog triggered (age={age:.2f}s).'
                    result.angle_turned = turned
                    self.get_logger().error(result.message)
                    return result

                turned = shortest_angle(yaw - yaw0)
                err = shortest_angle(target_angle - turned)

                if abs(err) <= self._angle_tol:
                    self._publish_stop()
                    goal_handle.succeed()
                    result.success = True
                    result.message = 'Reached target angle.'
                    result.angle_turned = turned
                    self.get_logger().info(
                        f'Rotate succeeded: turned {math.degrees(turned):+.2f} deg '
                        f'(target {math.degrees(target_angle):+.2f} deg).'
                    )
                    return result

                w = clamp(self._angular_kp * err, -speed_cap, speed_cap)
                cmd = Twist()
                cmd.angular.z = w
                self._cmd_pub.publish(cmd)

                fb = Rotate.Feedback()
                fb.angle_turned = turned
                fb.angle_remaining = err
                goal_handle.publish_feedback(fb)

                time.sleep(period)

            self._publish_stop()
            goal_handle.abort()
            result.success = False
            result.message = 'Shutting down.'
            result.angle_turned = turned
            return result
        finally:
            self._publish_stop()
            self._release_goal()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotionController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.get_logger().info('Shutting down motion_controller; sending zero Twist.')
            node._publish_stop()
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
