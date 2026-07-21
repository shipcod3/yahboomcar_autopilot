#!/bin/bash
# Start the Tesla-style vision+LiDAR+YOLO autopilot on the Yahboom MicroROS-Car-Pi5
# Usage: sh ~/start_autopilot.sh
#
# Auto-deploys fresh on every run:
#   - Copies package source + AI models into the running container
#   - Installs onnxruntime if missing (needed for YOLO)
#   - Builds the ROS2 package
#   - Launches the autopilot with two live preview windows
#
# Stop at any time with Ctrl+C.

set -e

PKG_SRC="/home/pi/yahboomcar_autopilot"

# Find the running ROS2 container
CONTAINER=$(docker ps --filter "ancestor=yahboomtechnology/ros-humble:4.1.2" --format "{{.Names}}" | head -1)

if [ -z "$CONTAINER" ]; then
    echo "ERROR: No running yahboomtechnology/ros-humble container found."
    echo "Please start the car's docker first (e.g.: sh ~/ros2_humble.sh)"
    exit 1
fi

echo "Using container: $CONTAINER"
echo ""

# ── Deploy package into container ──────────────────────────────────────────
echo "[1/4] Copying yahboomcar_autopilot into container..."
docker exec "$CONTAINER" bash -c "mkdir -p \
    /root/yahboomcar_ws/src/yahboomcar_autopilot/yahboomcar_autopilot \
    /root/yahboomcar_ws/src/yahboomcar_autopilot/launch \
    /root/yahboomcar_ws/src/yahboomcar_autopilot/resource \
    /root/yahboomcar_ws/src/yahboomcar_autopilot/models"

docker cp "$PKG_SRC/setup.py"    "$CONTAINER:/root/yahboomcar_ws/src/yahboomcar_autopilot/setup.py"
docker cp "$PKG_SRC/setup.cfg"   "$CONTAINER:/root/yahboomcar_ws/src/yahboomcar_autopilot/setup.cfg"
docker cp "$PKG_SRC/package.xml" "$CONTAINER:/root/yahboomcar_ws/src/yahboomcar_autopilot/package.xml"
docker cp "$PKG_SRC/yahboomcar_autopilot/." \
    "$CONTAINER:/root/yahboomcar_ws/src/yahboomcar_autopilot/yahboomcar_autopilot/"
docker cp "$PKG_SRC/launch/." \
    "$CONTAINER:/root/yahboomcar_ws/src/yahboomcar_autopilot/launch/"
docker exec "$CONTAINER" bash -c \
    "touch /root/yahboomcar_ws/src/yahboomcar_autopilot/resource/yahboomcar_autopilot"

# ── Copy AI models (YOLOv3-tiny + COCO labels) ─────────────────────────────
echo "[2/4] Copying AI models into container..."
if [ -d "$PKG_SRC/models" ]; then
    docker cp "$PKG_SRC/models/." \
        "$CONTAINER:/root/yahboomcar_ws/src/yahboomcar_autopilot/models/"
    echo "      Models copied."
else
    echo "      WARNING: $PKG_SRC/models not found - YOLO detection will be disabled."
fi

# ── Install Python deps (onnxruntime for YOLO) ─────────────────────────────
echo "[3/4] Checking Python dependencies..."
docker exec "$CONTAINER" bash -c "
    python3 -c 'import onnxruntime' 2>/dev/null || pip install onnxruntime -q
"

# ── Build ───────────────────────────────────────────────────────────────────
echo "[3/4] Building yahboomcar_autopilot..."
docker exec "$CONTAINER" bash -c "
    source /opt/ros/humble/setup.bash
    cd /root/yahboomcar_ws
    colcon build --packages-select yahboomcar_autopilot --symlink-install 2>&1 | tail -5
"

# ── Launch ──────────────────────────────────────────────────────────────────
echo "[4/4] Starting Tesla-style autopilot (camera + LiDAR + YOLO AI vision)..."
echo "      Windows: 'Autopilot Vision' (camera+YOLO) and 'LiDAR Map' (polar scan)"
echo "      Press Ctrl+C to stop."
echo ""

docker exec -it "$CONTAINER" bash -c "
source /opt/ros/humble/setup.bash
source /root/yahboomcar_ws/install/setup.bash
export ROS_DOMAIN_ID=20
ros2 run yahboomcar_autopilot vision_autopilot_node \
  --ros-args \
  -p linear_speed:=0.18 \
  -p max_angular_speed:=0.9 \
  -p lidar_emergency_stop_dist:=0.20 \
  -p lidar_slow_dist:=0.45 \
  -p yolo_cfg:=/root/yahboomcar_ws/src/yahboomcar_autopilot/models/yolov3-tiny.cfg \
  -p yolo_weights:=/root/yahboomcar_ws/src/yahboomcar_autopilot/models/yolov3-tiny.weights \
  -p yolo_names:=/root/yahboomcar_ws/src/yahboomcar_autopilot/models/coco.names \
  -p yolo_conf_thresh:=0.35 \
  -p yolo_input_size:=320 \
  -p yolo_detect_every_n:=3
"
