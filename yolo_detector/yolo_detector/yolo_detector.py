from __future__ import annotations

import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from std_msgs.msg import String
from ultralytics import YOLO

# tuned on our calibration frames
MIN_BLOB_AREA = 200
EPS_PIXELS = 50
MORPH_KERNEL = (5, 5)
PAD = 20
DARK_THRESH = 60
MIN_DARK_CONTOUR_AREA = 200

DETECT_RATE_HZ = 1.0


def preprocess_frame(img: np.ndarray) -> np.ndarray:
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
        bboxes.append((x, y, bw_w, bw_h))

    red_clusters: dict[int, list] = {}
    cluster_bboxes: dict[int, tuple[int, int, int, int]] = {}

    if points:
        pts = np.array(points)
        labels = DBSCAN(eps=EPS_PIXELS, min_samples=1).fit(pts).labels_
        for i, lbl in enumerate(labels):
            red_clusters.setdefault(int(lbl), []).append(bboxes[i])

    for lbl, members in red_clusters.items():
        xs, ys, x2s, y2s = [], [], [], []
        for bb in members:
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
    k_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for _lbl, (x_min, y_min, gw, gh) in cluster_bboxes.items():
        roi_gray = gray[y_min:y_min + gh, x_min:x_min + gw]
        roi_mask_red = mask_red[y_min:y_min + gh, x_min:x_min + gw]

        _, dark_mask = cv2.threshold(
            roi_gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV
        )
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_roi)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_roi)

        red_dil = cv2.dilate(roi_mask_red, k_roi, iterations=10)
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

    # cranked red channel so the network sees the same thing we trained on
    b, g, r = cv2.split(cv2.bitwise_and(img, img, mask=final_mask))
    r = cv2.add(r, 100)
    g = cv2.multiply(g, 0.5)
    b = cv2.multiply(b, 0.5)
    result_boost = cv2.merge([b, g, r])
    result_boost = cv2.bitwise_and(result_boost, result_boost, mask=final_mask)

    return result_boost


class YoloDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('yolo_detector')

        self.declare_parameter('weights_path', '')
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('min_confidence', 0.25)
        # label.train names -> what the follower code expects, 'left=turn' style
        self.declare_parameter(
            'class_aliases',
            [
                'Left Arrow=Turn',
                'Right Arrow=Turn',
                'Left Turn=Turn Around',
                'Right Turn=Turn Around',
            ],
        )

        weights_path = (
            self.get_parameter('weights_path').get_parameter_value().string_value
        )
        self._camera_device = (
            self.get_parameter('camera_device').get_parameter_value().string_value
        )
        self._min_conf = (
            self.get_parameter('min_confidence').get_parameter_value().double_value
        )

        raw_aliases = list(
            self.get_parameter('class_aliases')
            .get_parameter_value()
            .string_array_value
        )
        self._class_aliases: dict[str, str] = {}
        for entry in raw_aliases:
            if '=' not in entry:
                continue
            raw, canonical = entry.split('=', 1)
            raw = raw.strip()
            canonical = canonical.strip()
            if not raw or not canonical:
                continue
            self._class_aliases[raw] = canonical
        if self._class_aliases:
            self.get_logger().info('aliases: %r' % (self._class_aliases,))

        if not weights_path:
            wdir = os.path.abspath(os.path.dirname(__file__))
            got = None
            for _ in range(8):
                wdir = os.path.dirname(wdir)
                if not wdir or wdir == '/':
                    break
                cand = os.path.join(wdir, 'best.pt')
                if os.path.isfile(cand):
                    got = cand
                    break
            weights_path = got or os.path.join(os.getcwd(), 'best.pt')

        if not os.path.isfile(weights_path):
            raise FileNotFoundError('no weights at %r (pass weights_path?)' % (weights_path,))

        self.get_logger().info(f"Loading YOLO weights from '{weights_path}'.")
        self._model = YOLO(weights_path)

        # pi + laptop cams: open /dev/videoN by path + V4L2 first, indices are flaky
        attempts = []  # (opencv_source, backend, tag for log)
        if self._camera_device.startswith('/dev/video'):
            attempts.append((self._camera_device, cv2.CAP_V4L2, 'path+V4L2'))
            attempts.append((self._camera_device, cv2.CAP_ANY, 'path+ANY'))
            try:
                idx = int(self._camera_device.replace('/dev/video', ''))
                attempts.append((idx, cv2.CAP_V4L2, 'index+V4L2'))
            except ValueError:
                pass
        else:
            attempts.append((self._camera_device, cv2.CAP_ANY, 'source+ANY'))

        self._cap = None
        last_source = None
        for source, backend, label in attempts:
            self.get_logger().info("cam try %r %s" % (source, label))
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                self._cap = cap
                last_source = source
                break
            cap.release()

        if self._cap is None:
            raise RuntimeError(
                "opencv would not open %r (in use, wrong /dev, or not in `video` group?)"
                % (self._camera_device,)
            )
        self.get_logger().info('cam ok, source %r' % (last_source,))

        self._pub = self.create_publisher(String, 'yolo_detections', 10)

        period = 1.0 / max(DETECT_RATE_HZ, 0.1)
        self._timer = self.create_timer(period, self._on_tick)

        self.get_logger().info('spinning @ %.1fHz thr=%.2f' % (DETECT_RATE_HZ, self._min_conf))

    def _on_tick(self) -> None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Failed to read frame from camera.')
            return

        try:
            processed = preprocess_frame(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error('preprocess: %s' % (exc,))
            return

        img_h, img_w = processed.shape[:2]

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
                raw_cls_name = names.get(cls_id, str(cls_id)) if isinstance(
                    names, dict
                ) else str(cls_id)
                cls_name = self._class_aliases.get(raw_cls_name, raw_cls_name)

                x1, y1, x2, y2 = (
                    float(v.item()) for v in boxes.xyxy[i]
                )
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

                aliased = cls_name != raw_cls_name
                alias_hint = f" [raw='{raw_cls_name}']" if aliased else ''
                self.get_logger().info(
                    "det %s%s conf=%.2f box %.0f %.0f center %.0f %.0f"
                    % (cls_name, alias_hint, conf, bw, bh, cx, cy)
                )

                payload = {
                    'class': cls_name,
                    'raw_class': raw_cls_name,
                    'class_id': cls_id,
                    'conf': round(conf, 4),
                    'cx': round(cx, 1),
                    'cy': round(cy, 1),
                    'w': round(bw, 1),
                    'h': round(bh, 1),
                    'img_w': int(img_w),
                    'img_h': int(img_h),
                }
                msg = String()
                msg.data = json.dumps(payload)
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
