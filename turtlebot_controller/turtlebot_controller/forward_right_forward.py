"""Sequenced demo: drive 1 ft forward, turn right 90 deg, drive 1 ft forward.

Uses the ``/drive`` and ``/rotate`` action servers exposed by
``motion_controller``. Each step waits for the previous one to finish
before sending the next goal, so this works even though the controller
only allows one active goal at a time.

Run with:

    ros2 run turtlebot_controller forward_right_forward

Optional positional args:

    ros2 run turtlebot_controller forward_right_forward \\
        <linear_speed_mps> <angular_speed_rad_s>
"""

from __future__ import annotations

import math
import sys
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from turtlebot_controller_msgs.action import Drive, Rotate


FOOT_IN_METERS = 0.3048
DEFAULT_LINEAR_SPEED = 0.15
DEFAULT_ANGULAR_SPEED = 0.6
RIGHT_TURN_RAD = -math.pi / 2.0


class SequenceRunner(Node):
    """Send a Drive, then a Rotate, then a Drive; wait between each."""

    def __init__(self, linear_speed: float, angular_speed: float) -> None:
        super().__init__('forward_right_forward')
        self._linear_speed = float(linear_speed)
        self._angular_speed = float(angular_speed)

        self._drive_client = ActionClient(self, Drive, 'drive')
        self._rotate_client = ActionClient(self, Rotate, 'rotate')

    def _drive_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        self.get_logger().info(
            f'drive: {fb.distance_traveled:.3f} m traveled, '
            f'{fb.distance_remaining:.3f} m remaining @ {fb.current_speed:.2f} m/s'
        )

    def _rotate_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        self.get_logger().info(
            f'rotate: {math.degrees(fb.angle_turned):+.2f} deg turned, '
            f'{math.degrees(fb.angle_remaining):+.2f} deg remaining'
        )

    def _send_and_wait(self, client: ActionClient, goal, feedback_cb, label: str) -> bool:
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server for '{label}' not available.")
            return False

        send_future = client.send_goal_async(goal, feedback_callback=feedback_cb)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"{label} goal rejected.")
            return False
        self.get_logger().info(f'{label} goal accepted.')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().error(f'{label} did not return a result.')
            return False
        result = wrapped.result
        success = bool(getattr(result, 'success', False))
        message = getattr(result, 'message', '')
        self.get_logger().info(
            f"{label} finished: success={success}, message='{message}'"
        )
        return success

    def drive_one_foot(self) -> bool:
        goal = Drive.Goal()
        goal.distance = FOOT_IN_METERS
        goal.speed = self._linear_speed
        return self._send_and_wait(self._drive_client, goal, self._drive_feedback, 'Drive')

    def turn_right_90(self) -> bool:
        goal = Rotate.Goal()
        goal.angle = RIGHT_TURN_RAD
        goal.angular_speed = self._angular_speed
        return self._send_and_wait(
            self._rotate_client, goal, self._rotate_feedback, 'Rotate'
        )

    def run(self) -> bool:
        self.get_logger().info('Step 1/3: driving forward 1 ft (0.3048 m)...')
        if not self.drive_one_foot():
            return False

        self.get_logger().info('Step 2/3: turning right 90 deg...')
        if not self.turn_right_90():
            return False

        self.get_logger().info('Step 3/3: driving forward 1 ft (0.3048 m)...')
        if not self.drive_one_foot():
            return False

        self.get_logger().info('Sequence complete.')
        return True


def _parse_args(argv: list[str]) -> tuple[float, float]:
    linear = DEFAULT_LINEAR_SPEED
    angular = DEFAULT_ANGULAR_SPEED
    if len(argv) >= 1:
        try:
            linear = float(argv[0])
        except ValueError:
            print(
                'Usage: forward_right_forward [linear_speed_mps] [angular_speed_rad_s]',
                file=sys.stderr,
            )
            sys.exit(2)
    if len(argv) >= 2:
        try:
            angular = float(argv[1])
        except ValueError:
            print(
                'Usage: forward_right_forward [linear_speed_mps] [angular_speed_rad_s]',
                file=sys.stderr,
            )
            sys.exit(2)
    return linear, angular


def main(args=None) -> None:
    rclpy.init(args=args)

    linear_speed, angular_speed = _parse_args(sys.argv[1:])

    node = SequenceRunner(linear_speed, angular_speed)
    # Run the sequence on a background thread so Ctrl-C cleanly unwinds.
    success = False
    worker_exc: list[BaseException] = []

    def _worker() -> None:
        nonlocal success
        try:
            success = node.run()
        except BaseException as exc:  # noqa: BLE001 - surface to main thread
            worker_exc.append(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    try:
        worker.start()
        worker.join()
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C received; aborting sequence.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if worker_exc:
        raise worker_exc[0]
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
