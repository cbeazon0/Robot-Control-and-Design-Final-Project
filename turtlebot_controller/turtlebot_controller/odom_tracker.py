"""Process `/odom` and publish a cleaned 2D robot state + cumulative distance.

The TurtleBot base already publishes `nav_msgs/Odometry` on `/odom`. This node
does not add new information; it just extracts the pieces that are easy to
consume downstream (x, y, yaw in radians, cumulative driven distance) so that
other code does not have to repeat quaternion math.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return yaw (rotation about +Z) in radians from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomTracker(Node):
    def __init__(self) -> None:
        super().__init__('odom_tracker')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('state_topic', '/robot_state')
        self.declare_parameter('publish_rate_hz', 20.0)

        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pose_pub = self.create_publisher(Pose2D, state_topic, 10)
        self._dist_pub = self.create_publisher(Float64, '/cumulative_distance', 10)

        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, sensor_qos
        )

        self._last_x: float | None = None
        self._last_y: float | None = None
        self._cum_distance: float = 0.0
        self._latest_pose: Pose2D | None = None

        period = 1.0 / max(publish_rate, 1.0)
        self._timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            f"odom_tracker: subscribing to '{odom_topic}', publishing Pose2D on "
            f"'{state_topic}' and cumulative distance on '/cumulative_distance'."
        )

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if self._last_x is not None and self._last_y is not None:
            self._cum_distance += math.hypot(x - self._last_x, y - self._last_y)
        self._last_x = x
        self._last_y = y

        pose = Pose2D()
        pose.x = x
        pose.y = y
        pose.theta = yaw
        self._latest_pose = pose

    def _publish(self) -> None:
        if self._latest_pose is None:
            return
        self._pose_pub.publish(self._latest_pose)
        dist_msg = Float64()
        dist_msg.data = self._cum_distance
        self._dist_pub.publish(dist_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
