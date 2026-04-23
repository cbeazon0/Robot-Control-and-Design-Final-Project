"""YOLO detector node.

Reads frames from ``/dev/video0``, runs the red/quad preprocessing pipeline
(copied from the user's offline script), feeds the preprocessed frame into a
YOLO model loaded from a ``best.pt`` weights file, and logs each detection's
class name and confidence. Nothing is ever written to disk.

Dependencies (install with pip if missing):

    pip install ultralytics opencv-python scikit-learn numpy

Run with:

    ros2 run turtlebot_controller yolo_detector

Parameters (all optional):

    weights_path    : absolute path to the .pt file. If empty, the node
                      searches parent directories for 'src/best.pt'.
    camera_device   : OpenCV-compatible device (default '/dev/video0').
    detect_rate_hz  : how often to grab a frame and run inference.
    min_confidence  : drop detections below this score.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "ultralytics is not installed. Install with: pip install ultralytics"
    ) from exc


# Preprocessing parameters (match the user's offline script).
MIN_BLOB_AREA = 200
EPS_PIXELS = 50
MORPH_KERNEL = (5, 5)
PAD = 20
DARK_THRESH = 60
MIN_DARK_CONTOUR_AREA = 200


def preprocess_frame(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (boosted BGR image, binary final_mask).

    Keeps only red pixels that lie inside a dark quadrilateral, then pushes
    those pixels toward saturated red. Non-kept pixels are black.
    """
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 50])
    upper2 = np.array([179, 255, 255])

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    points: list[list[float]] = []
    valid_contours: list[np.ndarray] = []
    bboxes: list[tuple[int, int, int, int]] = []

    for c in contours:
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            continue
        m = cv2.moments(c)
        if m['m00'] == 0:
            continue
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
        x, y, bw_w, bw_h = cv2.boundingRect(c)
        points.append([cx, cy])
        valid_contours.append(c)
        bboxes.append((x, y, bw_w, bw_h))

    red_clusters: dict[int, list] = {}
    cluster_bboxes: dict[int, tuple[int, int, int, int]] = {}

    if points:
        pts = np.array(points)
        labels = DBSCAN(eps=EPS_PIXELS, min_samples=1).fit(pts).labels_
        for i, lbl in enumerate(labels):
            red_clusters.setdefault(int(lbl), []).append(
                (valid_contours[i], points[i], bboxes[i])
            )

    for lbl, members in red_clusters.items():
        xs, ys, x2s, y2s = [], [], [], []
        for _, _, bb in members:
            x, y, bw_w, bw_h = bb
            xs.append(x)
            ys.append(y)
            x2s.append(x + bw_w)
            y2s.append(y + bw_h)
        x_min = max(0, int(min(xs)) - PAD)
        y_min = max(0, int(min(ys)) - PAD)
        x_max = min(w - 1, int(max(x2s)) + PAD)
        y_max = min(h - 1, int(max(y2s)) + PAD)
        cluster_bboxes[lbl] = (x_min, y_min, x_max - x_min, y_max - y_min)

    quad_mask_global = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for _lbl, (x_min, y_min, gw, gh) in cluster_bboxes.items():
        roi_gray = gray[y_min:y_min + gh, x_min:x_min + gw]
        roi_mask_red = mask_red[y_min:y_min + gh, x_min:x_min + gw]

        _, dark_mask = cv2.threshold(
            roi_gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k)

        red_dil = cv2.dilate(roi_mask_red, k, iterations=10)
        focused_dark = cv2.bitwise_and(dark_mask, cv2.bitwise_not(red_dil))

        d_cnts, _ = cv2.findContours(
            focused_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        quads = []
        for c in d_cnts:
            if cv2.contourArea(c) < MIN_DARK_CONTOUR_AREA:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx)

        red_pts = np.column_stack(np.where(roi_mask_red > 0))
        if len(red_pts) == 0:
            continue

        cy, cx = np.mean(red_pts, axis=0)
        red_center = (int(cx), int(cy))

        best_quad = None
        best_score = -1
        for approx in quads:
            inside = cv2.pointPolygonTest(approx, red_center, False) >= 0
            quad_mask = np.zeros((gh, gw), dtype=np.uint8)
            cv2.drawContours(quad_mask, [approx], -1, 255, -1)
            red_inside = cv2.bitwise_and(
                roi_mask_red, roi_mask_red, mask=quad_mask
            )
            score = cv2.countNonZero(red_inside) + (100000 if inside else 0)
            if score > best_score:
                best_score = score
                best_quad = approx

        if best_quad is not None:
            best_quad_global = best_quad + np.array([[x_min, y_min]])
            cv2.drawContours(quad_mask_global, [best_quad_global], -1, 255, -1)

    final_mask = cv2.bitwise_and(mask_red, quad_mask_global)
    result = cv2.bitwise_and(img, img, mask=final_mask)

    # Natural red boost: pump R, suppress G and B, then re-mask to avoid leak.
    b, g, r = cv2.split(result)
    r = cv2.add(r, 100)
    g = cv2.multiply(g, 0.5)
    b = cv2.multiply(b, 0.5)
    result_boost = cv2.merge([b, g, r])
    result_boost = cv2.bitwise_and(result_boost, result_boost, mask=final_mask)

    return result_boost, final_mask


def _find_default_weights() -> str:
    """Walk up from this file looking for a 'src/best.pt'."""
    here = os.path.abspath(os.path.dirname(__file__))
    probe = here
    for _ in range(8):
        probe = os.path.dirname(probe)
        if not probe or probe == '/':
            break
        cand = os.path.join(probe, 'src', 'best.pt')
        if os.path.isfile(cand):
            return cand
    return os.path.join(os.getcwd(), 'src', 'best.pt')


class YoloDetectorNode(Node):
    """Grab frames, preprocess, run YOLO, log detections."""

    def __init__(self) -> None:
        super().__init__('yolo_detector')

        self.declare_parameter('weights_path', '')
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('detect_rate_hz', 5.0)
        self.declare_parameter('min_confidence', 0.25)

        weights_path = (
            self.get_parameter('weights_path').get_parameter_value().string_value
        )
        self._camera_device = (
            self.get_parameter('camera_device').get_parameter_value().string_value
        )
        rate_hz = (
            self.get_parameter('detect_rate_hz').get_parameter_value().double_value
        )
        self._min_conf = (
            self.get_parameter('min_confidence').get_parameter_value().double_value
        )

        if not weights_path:
            weights_path = _find_default_weights()

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path}'. "
                "Set the 'weights_path' parameter or place best.pt under "
                "your workspace's src/ directory."
            )

        self.get_logger().info(f"Loading YOLO weights from '{weights_path}'.")
        self._model = YOLO(weights_path)

        # VideoCapture accepts both a numeric index and a /dev/video* path.
        source: int | str = self._camera_device
        if self._camera_device.startswith('/dev/video'):
            try:
                source = int(self._camera_device.replace('/dev/video', ''))
            except ValueError:
                source = self._camera_device

        self.get_logger().info(
            f"Opening camera '{self._camera_device}' (cv2 source={source})."
        )
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera '{self._camera_device}'. "
                'Check permissions and that the device exists.'
            )

        self._pub = self.create_publisher(String, 'yolo_detections', 10)

        period = 1.0 / max(rate_hz, 0.1)
        self._timer = self.create_timer(period, self._on_tick)

        self.get_logger().info(
            f"yolo_detector ready: rate={rate_hz:.1f} Hz, "
            f"min_confidence={self._min_conf:.2f}."
        )

    def _on_tick(self) -> None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Failed to read frame from camera.')
            return

        try:
            processed, _ = preprocess_frame(frame)
        except Exception as exc:  # noqa: BLE001 - don't let one bad frame kill the node
            self.get_logger().error(f'Preprocessing failed: {exc}')
            return

        results = self._model.predict(
            source=processed,
            verbose=False,
            save=False,
            show=False,
        )

        for r in results:
            names = getattr(r, 'names', {}) or {}
            boxes = getattr(r, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                if conf < self._min_conf:
                    continue
                cls_id = int(boxes.cls[i].item())
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(
                    names, dict
                ) else str(cls_id)
                self.get_logger().info(
                    f"Detected '{cls_name}' (conf={conf:.3f})"
                )
                msg = String()
                msg.data = f'{cls_name}:{conf:.3f}'
                self._pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloDetectorNode()
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
