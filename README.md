# Yahboom MicroROS-Car-Pi5 — Tesla-Style Autopilot

Fully-autonomous, self-driving behaviour on top of the stock Yahboom
MicroROS-Car-Pi5 stack (ROS2 Humble). No remote control, no map needed.
Just put the car on the floor and it drives itself — detecting objects,
avoiding obstacles, and showing you everything it sees live on screen.

---

## 🚀 Quick Start (one command)

```bash
sh ~/start_autopilot.sh
```

That's it. The script does **everything automatically** on every run:

| Step | What happens |
|------|-------------|
| 1 | Finds the running Docker container |
| 2 | Copies the autopilot source code into the container |
| 3 | Copies the AI model files (YOLOv3-tiny) into the container |
| 4 | Installs onnxruntime if not already present |
| 5 | Builds the ROS2 package with colcon |
| 6 | Launches the autopilot with `ROS_DOMAIN_ID=20` |

> **Stop the car at any time** with `Ctrl+C` — the node sends a zero-velocity
> stop command before exiting.

> **Safe to run after every reboot** — works with whatever new container
> Docker creates; no manual setup required.

---

## ✨ Features

### 🤖 Autonomous Navigation ("Tesla mode")
- **Camera-based floor detection** — flood-fill algorithm detects drivable
  floor in real time using the front camera; no map, GPS, or depth sensor needed
- **9-column clearance steering** — measures open space in 9 vertical
  slices of the camera view; steers toward whichever direction has the
  most clear path ahead
- **Obstacle search** — when blocked, the car rotates in place to find a
  new opening; flips direction periodically if still stuck

### 🔴 LiDAR Safety
- **Emergency stop** — hard stops when the LiDAR detects any obstacle
  within 20 cm in the forward arc
- **Speed reduction** — slows proportionally as obstacles approach within
  45 cm (blends camera steering with LiDAR speed scaling)
- **Buzzer control** — one short beep when an e-stop first triggers,
  buzzer automatically silenced the moment the path clears

### 🧠 AI Object Detection (YOLOv3-tiny)
- Runs **80-class COCO object detection** on every 3rd camera frame
- Detects and labels in real time:
  - 🟥 **Danger** (red boxes): `person`, `cat`, `dog`, `chair`, `couch`,
    `dining table`, `bed`, `bicycle`, `car`, `motorbike`, `truck`, `bus`
  - 🟦 **Other** (cyan boxes): all remaining 68 COCO classes
- Shows **class name + confidence %** on each bounding box
- Uses **cv2.dnn** with the OpenCV CPU backend — no GPU needed, runs at
  ~3–5 fps detection rate on the Pi5

### 🖥️ Live Preview Windows

#### "Autopilot Vision" window
- Camera feed with green overlay showing detected drivable floor
- Red vertical bars showing per-column clearance depth
- Blue dot = floor seed point
- YOLO bounding boxes with labels overlay on top
- Status text: `DRIVING` / `SEARCHING` / `LIDAR E-STOP`
- LiDAR front distance readout
- YOLO object count

#### "LiDAR Map" window
- Polar top-down map of the LiDAR scan updated at 10 Hz
- Concentric range rings at 0.5 m, 1.0 m, 2.0 m, 3.0 m
- Points colour-coded: 🔴 red = very close, 🟢 green = safe distance
- Red inner circle = emergency stop zone (20 cm)
- Cyan outer circle = slow-down zone (45 cm)
- Car icon with forward arrow at centre
- Status: `DRIVE` / `SEARCH` / `E-STOP` + front distance

---

## 📁 File Structure

```
~/yahboomcar_autopilot/
├── yahboomcar_autopilot/
│   └── vision_autopilot_node.py   ← Main autopilot (camera + LiDAR + YOLO)
├── models/
│   ├── yolov3-tiny.cfg            ← YOLOv3-tiny network config
│   ├── yolov3-tiny.weights        ← YOLOv3-tiny weights (34 MB)
│   └── coco.names                 ← 80 COCO class labels
└── launch/
    └── vision_autopilot_launch.py ← Optional: launch inside container

~/start_autopilot.sh               ← THE ONE SCRIPT TO RUN EVERYTHING
```

---

## ⚙️ Tunable Parameters

All can be set via `ros2 run ... --ros-args -p name:=value`.
The defaults in `start_autopilot.sh` are tuned for indoor floor driving.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `linear_speed` | `0.18` m/s | Forward driving speed |
| `max_angular_speed` | `0.9` rad/s | Max turning speed |
| `lidar_emergency_stop_dist` | `0.20` m | Hard stop distance |
| `lidar_slow_dist` | `0.45` m | Start slowing distance |
| `camera_index` | `0` | Camera device index |
| `flip_horizontal` | `false` | Flip camera if mirrored |
| `stop_clearance_ratio` | `0.22` | Floor ratio before searching |
| `yolo_conf_thresh` | `0.35` | YOLO detection confidence |
| `yolo_input_size` | `320` | YOLO input resolution (smaller=faster) |
| `yolo_detect_every_n` | `3` | Run YOLO every N control cycles |
| `show_preview` | `true` | Show camera preview window |
| `show_lidar_preview` | `true` | Show LiDAR map window |

---

## 🛠️ Manual Setup (optional, not needed with start_autopilot.sh)

If you want to manually build and run inside the container:

```bash
# On the Pi host:
sh ~/start_agent_rpi5.sh      # Start micro-ROS agent (separate terminal)
sh ~/ros2_humble.sh           # Enter the car's docker container

# Inside the container:
pip install onnxruntime
cp -r /path/to/yahboomcar_autopilot ~/yahboomcar_ws/src/
cd ~/yahboomcar_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yahboomcar_autopilot --symlink-install
source install/setup.bash
export ROS_DOMAIN_ID=20
ros2 run yahboomcar_autopilot vision_autopilot_node
```

---

## 📋 Requirements

- Yahboom MicroROS-Car-Pi5 with the stock Docker setup
- USB/Astra RGB camera connected (camera index 0)
- YDLIDAR or compatible LiDAR (optional but recommended for safety)
- Pi5 with at least 2 GB RAM free
- Internet access on first run (to install onnxruntime ~15 MB)
- Model files pre-downloaded to `~/yahboomcar_autopilot/models/`

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Package 'yahboomcar_autopilot' not found` | Run `sh ~/start_autopilot.sh` — it auto-builds |
| Camera window not appearing | Make sure `DISPLAY=:0` is set in the container |
| YOLO not detecting | Check models exist: `ls ~/yahboomcar_autopilot/models/` |
| Car drives but no window | Run directly with `docker exec -it` (not `-d`) |
| Buzzer keeps beeping | Car is too close to an obstacle; move it away |
| Car won't move | Check LiDAR — if obstacle within 20 cm it will stay stopped |
