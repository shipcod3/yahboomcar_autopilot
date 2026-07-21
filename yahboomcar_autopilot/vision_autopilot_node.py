#!/usr/bin/env python3
"""
vision_autopilot_node : Camera + LiDAR + YOLO AI vision self-driving
for the Yahboom MicroROS-Car-Pi5.

Features:
  - YOLOv3-tiny object detection (cv2.dnn) - persons, chairs, tables, etc.
  - Flood-fill floor detection + clearance-based steering
  - Live "Autopilot Vision" preview with YOLO bounding boxes + labels
  - Live "LiDAR Map" polar preview
  - LiDAR e-stop + buzzer control
"""

import math
import os
import time

import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

COCO_NAMES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

DANGER_CLASSES = {"person","cat","dog","chair","couch","dining table","bed",
                  "refrigerator","toilet","bicycle","motorbike","car","truck",
                  "bus","potted plant"}

def _box_colour(label):
    return (0,50,255) if label in DANGER_CLASSES else (0,200,255)


class VisionAutopilotNode(Node):
    def __init__(self):
        super().__init__("vision_autopilot_node")

        self.declare_parameter("camera_index", 0)
        self.declare_parameter("flip_horizontal", False)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("linear_speed", 0.18)
        self.declare_parameter("max_angular_speed", 0.9)
        self.declare_parameter("num_columns", 9)
        self.declare_parameter("center_columns", 3)
        self.declare_parameter("stop_clearance_ratio", 0.22)
        self.declare_parameter("slow_clearance_ratio", 0.45)
        self.declare_parameter("search_angular_speed", 0.6)
        self.declare_parameter("search_timeout_sec", 4.0)
        self.declare_parameter("lo_diff", 12)
        self.declare_parameter("up_diff", 30)
        self.declare_parameter("seed_row_ratio", 0.94)
        self.declare_parameter("seed_half_width_ratio", 0.12)
        self.declare_parameter("roi_top_ratio", 0.35)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("show_preview", True)
        self.declare_parameter("show_lidar_preview", True)
        self.declare_parameter("lidar_preview_size", 400)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("lidar_emergency_stop_dist", 0.20)
        self.declare_parameter("lidar_emergency_angle_deg", 30.0)
        self.declare_parameter("lidar_slow_dist", 0.45)
        self.declare_parameter("yolo_cfg",
            "/root/yahboomcar_ws/src/yahboomcar_autopilot/models/yolov3-tiny.cfg")
        self.declare_parameter("yolo_weights",
            "/root/yahboomcar_ws/src/yahboomcar_autopilot/models/yolov3-tiny.weights")
        self.declare_parameter("yolo_names",
            "/root/yahboomcar_ws/src/yahboomcar_autopilot/models/coco.names")
        self.declare_parameter("yolo_conf_thresh", 0.35)
        self.declare_parameter("yolo_nms_thresh", 0.4)
        self.declare_parameter("yolo_input_size", 320)
        self.declare_parameter("yolo_detect_every_n", 3)

        g = self.get_parameter
        self.camera_index         = int(g("camera_index").value)
        self.flip_horizontal      = bool(g("flip_horizontal").value)
        self.control_rate_hz      = float(g("control_rate_hz").value)
        self.linear_speed         = float(g("linear_speed").value)
        self.max_angular_speed    = float(g("max_angular_speed").value)
        self.num_columns          = int(g("num_columns").value)
        self.center_columns       = int(g("center_columns").value)
        self.stop_clearance_ratio = float(g("stop_clearance_ratio").value)
        self.slow_clearance_ratio = float(g("slow_clearance_ratio").value)
        self.search_angular_speed = float(g("search_angular_speed").value)
        self.search_timeout_sec   = float(g("search_timeout_sec").value)
        self.lo_diff              = int(g("lo_diff").value)
        self.up_diff              = int(g("up_diff").value)
        self.seed_row_ratio       = float(g("seed_row_ratio").value)
        self.seed_half_width_ratio= float(g("seed_half_width_ratio").value)
        self.roi_top_ratio        = float(g("roi_top_ratio").value)
        self.publish_debug_image  = bool(g("publish_debug_image").value)
        self.show_preview         = bool(g("show_preview").value)
        self.show_lidar_preview   = bool(g("show_lidar_preview").value)
        self.lidar_preview_size   = int(g("lidar_preview_size").value)
        cmd_vel_topic             = g("cmd_vel_topic").value
        scan_topic                = g("scan_topic").value
        self.lidar_emergency_stop_dist = float(g("lidar_emergency_stop_dist").value)
        self.lidar_emergency_angle_deg = float(g("lidar_emergency_angle_deg").value)
        self.lidar_slow_dist      = float(g("lidar_slow_dist").value)
        yolo_cfg     = g("yolo_cfg").value
        yolo_weights = g("yolo_weights").value
        yolo_names   = g("yolo_names").value
        self.yolo_conf  = float(g("yolo_conf_thresh").value)
        self.yolo_nms   = float(g("yolo_nms_thresh").value)
        self.yolo_size  = int(g("yolo_input_size").value)
        self.yolo_every = int(g("yolo_detect_every_n").value)

        self.cmd_pub  = self.create_publisher(Twist, cmd_vel_topic, 5)
        self.beep_pub = self.create_publisher(UInt16, "/beep", 1)
        self.debug_pub = None
        self.bridge = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, "/vision_autopilot/debug_image", 1)
            self.bridge = CvBridge()

        self.capture = cv.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            self.get_logger().error("Could not open camera index %d." % self.camera_index)
        else:
            self.get_logger().info("Camera %d opened." % self.camera_index)

        self.yolo_net = None
        self.yolo_out_names = []
        self.class_names = COCO_NAMES
        self._load_yolo(yolo_cfg, yolo_weights, yolo_names)
        self._yolo_frame_cnt = 0
        self._last_detections = []

        self.searching = False
        self.search_start_time = None
        self.search_direction = 1.0
        self.lidar_estop = False
        self.lidar_min_front = float("inf")
        self.lidar_available = False
        self._last_scan = None

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 5)
        period = 1.0 / max(self.control_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.control_loop)
        self._lidar_timer = self.create_timer(0.1, self._update_lidar_preview)
        self.get_logger().info("vision_autopilot_node started (camera + LiDAR + YOLO).")

    def _load_yolo(self, cfg_path, weights_path, names_path):
        if not os.path.isfile(cfg_path):
            self.get_logger().warn("YOLO cfg not found: %s (detection disabled)" % cfg_path)
            return
        if not os.path.isfile(weights_path):
            self.get_logger().warn("YOLO weights not found: %s (detection disabled)" % weights_path)
            return
        try:
            self.yolo_net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            layer_names = self.yolo_net.getLayerNames()
            out_idx = self.yolo_net.getUnconnectedOutLayers()
            if hasattr(out_idx, "flatten"):
                out_idx = out_idx.flatten()
            self.yolo_out_names = [layer_names[i - 1] for i in out_idx]
            if os.path.isfile(names_path):
                with open(names_path) as f:
                    names = [l.strip() for l in f if l.strip()]
                if names:
                    self.class_names = names
            self.get_logger().info("YOLOv3-tiny loaded (%d classes, %dx%d)." %
                                   (len(self.class_names), self.yolo_size, self.yolo_size))
        except Exception as e:
            self.get_logger().error("Failed to load YOLO: %s" % str(e))
            self.yolo_net = None

    def _run_yolo(self, frame):
        if self.yolo_net is None:
            return []
        h, w = frame.shape[:2]
        sz = self.yolo_size
        blob = cv.dnn.blobFromImage(frame, 1/255.0, (sz, sz), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        try:
            outs = self.yolo_net.forward(self.yolo_out_names)
        except Exception:
            return []
        boxes, confs, class_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf < self.yolo_conf:
                    continue
                cx = int(det[0] * w); cy = int(det[1] * h)
                bw = int(det[2] * w); bh = int(det[3] * h)
                boxes.append([cx - bw // 2, cy - bh // 2, bw, bh])
                confs.append(conf); class_ids.append(cid)
        idxs = cv.dnn.NMSBoxes(boxes, confs, self.yolo_conf, self.yolo_nms)
        results = []
        if len(idxs):
            flat = idxs.flatten() if hasattr(idxs, "flatten") else idxs
            for i in flat:
                x, y, bw, bh = boxes[i]
                label = self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else str(class_ids[i])
                results.append((x, y, x + bw, y + bh, label, confs[i]))
        return results

    def _draw_detections(self, img, dets):
        for (x1, y1, x2, y2, label, conf) in dets:
            col = _box_colour(label)
            cv.rectangle(img, (x1, y1), (x2, y2), col, 2)
            txt = "%s %.0f%%" % (label, conf * 100)
            (tw, th), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), col, -1)
            cv.putText(img, txt, (x1 + 2, y1 - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            return
        self._last_scan = msg
        if not self.lidar_available:
            self.lidar_available = True
            self.get_logger().info("LiDAR data received.")
        n = len(msg.ranges)
        limit_rad = math.radians(self.lidar_emergency_angle_deg)
        min_front = float("inf")
        for i in range(n):
            angle = math.atan2(math.sin(msg.angle_min + i * msg.angle_increment),
                               math.cos(msg.angle_min + i * msg.angle_increment))
            if abs(angle) <= limit_rad:
                r = msg.ranges[i]
                if not math.isnan(r) and not math.isinf(r) and r > 0.01:
                    min_front = min(min_front, r)
        self.lidar_min_front = min_front
        if min_front < self.lidar_emergency_stop_dist:
            if not self.lidar_estop:
                self.get_logger().warn("LIDAR E-STOP: %.2fm!" % min_front)
                self.beep_pub.publish(UInt16(data=1))
            self.lidar_estop = True
            self.publish_cmd(0.0, 0.0)
        else:
            if self.lidar_estop:
                self.beep_pub.publish(UInt16(data=0))
            self.lidar_estop = False

    def _update_lidar_preview(self):
        if not self.show_lidar_preview or self._last_scan is None:
            return
        msg = self._last_scan
        sz = self.lidar_preview_size; cx = cy = sz // 2
        max_range = max(msg.range_max, 0.1)
        scale = (sz // 2 - 10) / max_range
        img = np.zeros((sz, sz, 3), dtype=np.uint8)
        for r_m in [0.5, 1.0, 2.0, 3.0]:
            r_px = int(r_m * scale)
            if r_px < sz // 2:
                cv.circle(img, (cx, cy), r_px, (40,40,40), 1)
                cv.putText(img, "%.1fm"%r_m, (cx+r_px+2,cy), cv.FONT_HERSHEY_SIMPLEX, 0.3, (60,60,60), 1)
        cv.line(img, (cx, cy), (cx, cy - sz//2 + 5), (0,60,0), 1)
        cv.putText(img, "FWD", (cx-12, 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0,100,0), 1)
        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            px = int(cx + r * scale * math.sin(angle))
            py = int(cy - r * scale * math.cos(angle))
            if 0 <= px < sz and 0 <= py < sz:
                ratio = min(r / max(self.lidar_slow_dist, 0.01), 1.0)
                cv.circle(img, (px, py), 2, (0, int(255*ratio), int(255*(1-ratio))), -1)
        cv.circle(img, (cx,cy), int(self.lidar_emergency_stop_dist*scale), (0,0,200), 1)
        cv.circle(img, (cx,cy), int(self.lidar_slow_dist*scale), (0,140,200), 1)
        cv.rectangle(img, (cx-8,cy-12), (cx+8,cy+12), (180,180,180), -1)
        cv.arrowedLine(img, (cx,cy+8), (cx,cy-14), (255,255,255), 2, tipLength=0.4)
        status = "E-STOP" if self.lidar_estop else ("SEARCH" if self.searching else "DRIVE")
        cv.putText(img, "LiDAR  "+status, (4,sz-6), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        if self.lidar_available:
            cv.putText(img, "front: %.2fm"%self.lidar_min_front, (4,sz-20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)
        try:
            cv.imshow("LiDAR Map", img); cv.waitKey(1)
        except Exception:
            pass

    def destroy_node(self):
        try: self.cmd_pub.publish(Twist())
        except Exception: pass
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        try: cv.destroyAllWindows()
        except Exception: pass
        super().destroy_node()

    def compute_floor_clearance(self, frame):
        h, w = frame.shape[:2]
        blurred = cv.GaussianBlur(frame, (7,7), 0)
        roi_top = int(h * self.roi_top_ratio)
        roi = blurred[roi_top:h, :]
        rh, rw = roi.shape[:2]
        seed_y = min(max(int(rh * self.seed_row_ratio), 0), rh - 1)
        seed_x = rw // 2
        mask = np.zeros((rh + 2, rw + 2), np.uint8)
        try:
            cv.floodFill(roi.copy(), mask, (seed_x, seed_y), 0,
                         loDiff=(self.lo_diff,)*3, upDiff=(self.up_diff,)*3,
                         flags=cv.FLOODFILL_MASK_ONLY|cv.FLOODFILL_FIXED_RANGE|(255<<8))
        except cv.error:
            return None, None
        floor_mask = mask[1:-1, 1:-1]
        col_edges = np.linspace(0, rw, self.num_columns + 1).astype(int)
        clearance = np.zeros(self.num_columns, dtype=np.float32)
        for c in range(self.num_columns):
            x0, x1 = col_edges[c], col_edges[c+1]
            col_is_floor = np.any(floor_mask[:, x0:x1] > 0, axis=1)
            count = 0
            for row in range(rh-1, -1, -1):
                if col_is_floor[row]: count += 1
                else: break
            clearance[c] = count
        debug_img = None
        if self.publish_debug_image or self.show_preview:
            debug_img = frame.copy()
            overlay = np.zeros_like(roi)
            overlay[floor_mask > 0] = (0, 200, 0)
            debug_img[roi_top:h, :] = cv.addWeighted(roi, 0.7, overlay, 0.3, 0)
            for c in range(self.num_columns):
                cx_col = (col_edges[c] + col_edges[c+1]) // 2
                top_y = roi_top + max(rh-1-int(clearance[c]), 0)
                cv.line(debug_img, (cx_col, h-1), (cx_col, top_y), (0,0,255), 2)
            cv.circle(debug_img, (seed_x, roi_top+seed_y), 5, (255,0,0), -1)
            status = "SEARCHING" if self.searching else "DRIVING"
            if self.lidar_estop: status = "LIDAR E-STOP"
            lidar_txt = ("LiDAR: %.2fm"%self.lidar_min_front) if self.lidar_available else "LiDAR: N/A"
            yolo_txt = "YOLO: ON (%d objs)" % len(self._last_detections) if self.yolo_net else "YOLO: model missing"
            cv.putText(debug_img, status,    (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv.putText(debug_img, lidar_txt, (10,50), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,200,0), 1)
            cv.putText(debug_img, yolo_txt,  (10,72), cv.FONT_HERSHEY_SIMPLEX, 0.45, (180,255,180), 1)
        return clearance / float(rh), debug_img

    def control_loop(self):
        if self.lidar_estop:
            self.publish_cmd(0.0, 0.0); return
        if self.capture is None or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.get_logger().warn("Camera read failed.", throttle_duration_sec=5.0); return
        if self.flip_horizontal:
            frame = cv.flip(frame, 1)
        self._yolo_frame_cnt += 1
        if self._yolo_frame_cnt >= self.yolo_every:
            self._yolo_frame_cnt = 0
            self._last_detections = self._run_yolo(frame)
        clearance, debug_img = self.compute_floor_clearance(frame)
        if debug_img is not None:
            self._draw_detections(debug_img, self._last_detections)
            if self.debug_pub is not None:
                try: self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))
                except Exception: pass
            if self.show_preview:
                try: cv.imshow("Autopilot Vision", debug_img); cv.waitKey(1)
                except Exception: pass
        if clearance is None:
            self.publish_cmd(0.0, 0.0); return
        n = self.num_columns
        center_lo = (n - self.center_columns) // 2
        center_clearance = float(np.min(clearance[center_lo:center_lo+self.center_columns]))
        best_col = int(np.argmax(clearance))
        now = time.time()
        if center_clearance < self.stop_clearance_ratio:
            if not self.searching:
                self.searching = True; self.search_start_time = now
                self.search_direction = 1.0 if best_col >= n/2.0 else -1.0
                self.get_logger().info("Autopilot: blocked ahead, searching for an opening.")
            self.publish_cmd(0.0, self.search_angular_speed * self.search_direction)
            if self.search_start_time and (now - self.search_start_time) > self.search_timeout_sec:
                self.search_direction *= -1.0; self.search_start_time = now
            return
        self.searching = False
        offset = (best_col - (n-1)/2.0) / ((n-1)/2.0)
        angular = -offset * self.max_angular_speed
        speed_scale = float(np.clip(
            (center_clearance - self.stop_clearance_ratio) /
            max(self.slow_clearance_ratio - self.stop_clearance_ratio, 1e-3), 0.15, 1.0))
        linear = self.linear_speed * speed_scale
        if self.lidar_available and self.lidar_min_front < self.lidar_slow_dist:
            lidar_scale = max(0.0, (self.lidar_min_front - self.lidar_emergency_stop_dist) /
                             max(self.lidar_slow_dist - self.lidar_emergency_stop_dist, 0.01))
            linear *= lidar_scale
        self.publish_cmd(linear, angular)

    def publish_cmd(self, linear, angular):
        t = Twist(); t.linear.x = float(linear); t.angular.z = float(angular)
        self.cmd_pub.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = VisionAutopilotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
