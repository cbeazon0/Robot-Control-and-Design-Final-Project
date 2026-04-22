"""Command-line helpers for the motion controller action servers.

Installed as three console scripts:

* ``ros2 run turtlebot_controller drive_cli <distance_m> [speed]``
  - ``<distance_m>``: meters; ``0`` (or omit) means "drive until Ctrl-C".
* ``ros2 run turtlebot_controller rotate_cli <angle_deg> [angular_speed]``
  - ``<angle_deg>``: signed degrees (``90``, ``-90``, ``180``, ...).
* ``ros2 run turtlebot_controller stop_cli``
  - Cancels any active Drive or Rotate goal on this controller.
"""

from __future__ import annotations

import math
import sys
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from turtlebot_controller_msgs.action import Drive, Rotate


class _DriveClient(Node):
    def __init__(self, distance: float, speed: float) -> None:
        super().__init__('drive_cli')
        self._distance = distance
        self._speed = speed
        self._client = ActionClient(self, Drive, 'drive')
        self._done = threading.Event()
        self._goal_handle = None
        self._result_success = False

    def _on_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        if self._distance > 0.0:
            self.get_logger().info(
                f'drive: {fb.distance_traveled:.3f} m traveled, '
                f'{fb.distance_remaining:.3f} m remaining @ {fb.current_speed:.2f} m/s'
            )
        else:
            self.get_logger().info(
                f'drive: {fb.distance_traveled:.3f} m traveled @ {fb.current_speed:.2f} m/s'
            )

    def send(self) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Drive action server '/drive' not available.")
            return False

        goal = Drive.Goal()
        goal.distance = float(self._distance)
        goal.speed = float(self._speed)

        send_future = self._client.send_goal_async(goal, feedback_callback=self._on_feedback)
        send_future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future) -> None:
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Drive goal rejected.')
            self._done.set()
            return
        self.get_logger().info('Drive goal accepted.')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        result = future.result().result
        self._result_success = bool(result.success)
        self.get_logger().info(
            f"Drive finished: success={result.success}, "
            f"traveled={result.distance_traveled:.3f} m, message='{result.message}'"
        )
        self._done.set()

    def cancel(self) -> None:
        if self._goal_handle is not None:
            self.get_logger().info('Canceling Drive goal...')
            self._goal_handle.cancel_goal_async()


class _RotateClient(Node):
    def __init__(self, angle_rad: float, angular_speed: float) -> None:
        super().__init__('rotate_cli')
        self._angle = angle_rad
        self._angular_speed = angular_speed
        self._client = ActionClient(self, Rotate, 'rotate')
        self._done = threading.Event()
        self._goal_handle = None
        self._result_success = False

    def _on_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        self.get_logger().info(
            f'rotate: {math.degrees(fb.angle_turned):+.2f} deg turned, '
            f'{math.degrees(fb.angle_remaining):+.2f} deg remaining'
        )

    def send(self) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Rotate action server '/rotate' not available.")
            return False

        goal = Rotate.Goal()
        goal.angle = float(self._angle)
        goal.angular_speed = float(self._angular_speed)

        send_future = self._client.send_goal_async(goal, feedback_callback=self._on_feedback)
        send_future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future) -> None:
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Rotate goal rejected.')
            self._done.set()
            return
        self.get_logger().info('Rotate goal accepted.')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        result = future.result().result
        self._result_success = bool(result.success)
        self.get_logger().info(
            f"Rotate finished: success={result.success}, "
            f"turned={math.degrees(result.angle_turned):+.2f} deg, message='{result.message}'"
        )
        self._done.set()

    def cancel(self) -> None:
        if self._goal_handle is not None:
            self.get_logger().info('Canceling Rotate goal...')
            self._goal_handle.cancel_goal_async()


def _spin_until_done(node: Node, done_event: threading.Event, canceler) -> None:
    executor_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    executor_thread.start()
    try:
        while not done_event.is_set():
            done_event.wait(timeout=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C received; canceling goal...')
        canceler()
        done_event.wait(timeout=5.0)
    finally:
        pass


def drive_main(args=None) -> None:
    rclpy.init(args=args)

    argv = sys.argv[1:]
    if not argv:
        distance = 0.0
    else:
        try:
            distance = float(argv[0])
        except ValueError:
            print('Usage: drive_cli <distance_m> [speed]', file=sys.stderr)
            rclpy.shutdown()
            sys.exit(2)

    speed = 0.15
    if len(argv) >= 2:
        try:
            speed = float(argv[1])
        except ValueError:
            print('Usage: drive_cli <distance_m> [speed]', file=sys.stderr)
            rclpy.shutdown()
            sys.exit(2)

    node = _DriveClient(distance, speed)
    try:
        if not node.send():
            return
        _spin_until_done(node, node._done, node.cancel)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def rotate_main(args=None) -> None:
    rclpy.init(args=args)

    argv = sys.argv[1:]
    if not argv:
        print('Usage: rotate_cli <angle_deg> [angular_speed_rad_s]', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(2)

    try:
        angle_deg = float(argv[0])
    except ValueError:
        print('Usage: rotate_cli <angle_deg> [angular_speed_rad_s]', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(2)

    angular_speed = 0.6
    if len(argv) >= 2:
        try:
            angular_speed = float(argv[1])
        except ValueError:
            print('Usage: rotate_cli <angle_deg> [angular_speed_rad_s]', file=sys.stderr)
            rclpy.shutdown()
            sys.exit(2)

    node = _RotateClient(math.radians(angle_deg), angular_speed)
    try:
        if not node.send():
            return
        _spin_until_done(node, node._done, node.cancel)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def stop_main(args=None) -> None:
    """Cancel every active Drive and Rotate goal on this controller.

    Calls the action-level ``_action/cancel_goal`` service on both servers
    with an empty request, which cancels all goals on each.
    """
    from action_msgs.srv import CancelGoal

    rclpy.init(args=args)
    node = rclpy.create_node('stop_cli')
    try:
        for action_name in ('drive', 'rotate'):
            srv_name = f'/{action_name}/_action/cancel_goal'
            client = node.create_client(CancelGoal, srv_name)
            if not client.wait_for_service(timeout_sec=2.0):
                node.get_logger().warn(
                    f"Cancel service '{srv_name}' not available; skipping."
                )
                continue
            future = client.call_async(CancelGoal.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
            response = future.result()
            if response is None:
                node.get_logger().warn(f"No response from '{srv_name}'.")
                continue
            node.get_logger().info(
                f"'/{action_name}': canceled {len(response.goals_canceling)} goal(s) "
                f'(return_code={response.return_code}).'
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
