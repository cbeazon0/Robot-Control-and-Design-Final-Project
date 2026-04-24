from __future__ import annotations

import json
import os

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

# inference timer (grabs + runs YOLO at this rate)
DETECT_RATE_HZ = 1.0


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

        # OpenCV BGR, unmodified — must match what you trained on (camera / resize in YOLO)
        img_h, img_w = frame.shape[:2]

        results = self._model.predict(
            source=frame,
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
