# Robot-Control-and-Design-Final-Project

A ROS2 closed-loop motion controller for a TurtleBot. Exposes two actions,
`Drive` and `Rotate`, backed by a simple odometry-based P-controller.

## What's in here

Two ROS2 packages:

- `turtlebot_controller_msgs/` (ament_cmake) — defines the `Drive` and
  `Rotate` action interfaces.
- `turtlebot_controller/` (ament_python) — the actual nodes:
  - `odom_tracker` — subscribes to `/odom` and republishes a cleaned 2D
    state (`geometry_msgs/Pose2D` on `/robot_state`) plus cumulative
    distance (`std_msgs/Float64` on `/cumulative_distance`).
  - `motion_controller` — hosts the `/drive` and `/rotate` action servers
    and publishes `geometry_msgs/Twist` on `/cmd_vel`.
  - `drive_cli`, `rotate_cli`, `stop_cli` — convenience command-line
    clients.

The TurtleBot base driver already publishes `/odom`, so this project only
**consumes** odometry; it does not generate it.

## Actions

`turtlebot_controller_msgs/action/Drive`:

```
# Goal
float64 distance        # meters. 0 or negative = drive forward until canceled
float64 speed           # m/s, positive, clamped to max
---
# Result
float64 distance_traveled
bool    success
string  message
---
# Feedback
float64 distance_traveled
float64 distance_remaining
float64 current_speed
```

`turtlebot_controller_msgs/action/Rotate`:

```
# Goal
float64 angle           # radians, signed. +pi/2 = 90 deg CCW, -pi/2 = 90 deg CW
float64 angular_speed   # rad/s, positive, clamped to max
---
# Result
float64 angle_turned
bool    success
string  message
---
# Feedback
float64 angle_turned
float64 angle_remaining
```

## Build

From the root of your ROS2 workspace (one that contains these two packages
under `src/`):

```bash
cd ~/ros2_ws
colcon build --packages-select turtlebot_controller_msgs turtlebot_controller
source install/setup.bash
```

If you cloned this repo directly into a workspace's `src/`, replace the
path above accordingly.

## Run

Bring the robot up (TurtleBot3 example — use whatever bringup matches your
hardware):

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

Then launch the controller:

```bash
ros2 launch turtlebot_controller bringup.launch.py
```

This starts the `odom_tracker` and `motion_controller` nodes with the
defaults from
[turtlebot_controller/config/params.yaml](turtlebot_controller/config/params.yaml).

### Send goals

Using the built-in `ros2 action` CLI:

```bash
# Drive 1.5 m forward at up to 0.2 m/s
ros2 action send_goal /drive turtlebot_controller_msgs/action/Drive \
    "{distance: 1.5, speed: 0.2}" --feedback

# Drive forward indefinitely at 0.15 m/s (Ctrl-C to cancel)
ros2 action send_goal /drive turtlebot_controller_msgs/action/Drive \
    "{distance: 0.0, speed: 0.15}" --feedback

# Turn 90 deg CCW
ros2 action send_goal /rotate turtlebot_controller_msgs/action/Rotate \
    "{angle: 1.5707963, angular_speed: 0.6}" --feedback

# Turn 90 deg CW
ros2 action send_goal /rotate turtlebot_controller_msgs/action/Rotate \
    "{angle: -1.5707963, angular_speed: 0.6}" --feedback
```

Using the convenience CLIs shipped with this package:

```bash
# 1.5 m forward at default speed
ros2 run turtlebot_controller drive_cli 1.5

# Drive until Ctrl-C
ros2 run turtlebot_controller drive_cli

# Turn 90 deg CCW / CW (degrees!)
ros2 run turtlebot_controller rotate_cli 90
ros2 run turtlebot_controller rotate_cli -90

# Cancel any active goal on /drive and /rotate
ros2 run turtlebot_controller stop_cli
```

## Tuning and safety notes

- The controller rejects a new goal while another is active, so `Drive`
  and `Rotate` cannot fight over `/cmd_vel`.
- If `/odom` stops arriving for longer than `odom_timeout_s` (default
  0.5 s), the active goal is aborted and a zero `Twist` is published.
- A zero `Twist` is also published on node shutdown.
- Start with conservative limits on real hardware (defaults:
  `max_linear_speed=0.15 m/s`, `max_angular_speed=0.6 rad/s`) and tune
  `linear_kp` / `angular_kp` and the tolerances to taste.
- TurtleBot wheel odometry drifts with slip; long-range absolute distance
  error is inherent to closed-loop odometry control.
- If your platform publishes odometry on a different topic (for example
  `/odometry/filtered` on some bringups), override the `odom_topic`
  parameter in [config/params.yaml](turtlebot_controller/config/params.yaml).
