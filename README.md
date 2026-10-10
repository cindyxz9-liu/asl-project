# Insta360 ASL stream with camera telemetry and a HaWoR-style world-space pipeline

Live hand tracking from an Insta360 (tested with the Ace Pro 2 in Webcam Mode),
with camera telemetry and a world-space hand reconstruction pipeline modelled
on HaWoR (Zhang et al., CVPR 2025, https://github.com/ThunderVVV/HaWoR).

## Run

```
python3 asl_stream.py                       # live camera, static telemetry (desk-mounted rig)
python3 asl_stream.py --telemetry visual    # live camera, ego-motion estimated from the video
python3 asl_stream.py --video clip.insv     # recording: gyro/accel telemetry read from the file
python3 asl_stream.py --video clip.mp4 --telemetry visual
```

Options: `--hfov` (horizontal field of view, default 100 degrees) or `--focal`
(pixels) set the pinhole intrinsics; `--out` sets the JSONL path. The script
refuses any camera whose name is not the Insta360 unless `--allow-any-camera`
is given. With `--video` and no `--telemetry`, the file is probed for IMU data
and falls back to `visual` with a printed message if there is none (or if
telemetry-parser is not installed).

Dependencies on top of the original script: `scipy`, and `telemetry-parser`
for `.insv` files (`python3 -m pip install --user telemetry-parser`).

## Telemetry sources (`telemetry.py`)

| source   | where it comes from                                  | rotation | translation     |
|----------|------------------------------------------------------|----------|-----------------|
| `static` | nothing; camera assumed fixed                        | identity | zero            |
| `visual` | ORB features on the background, hands masked out     | metric   | direction only  |
| `insv`   | gyro + accelerometer embedded in an Insta360 recording | metric, gravity-aligned | zero |

Webcam Mode streams plain UVC video and carries no IMU data, so `insv` only
works on files recorded by the camera itself. `visual` fits both a homography
(pure rotation) and an essential matrix (rotation + translation) per keyframe
and keeps whichever explains more matches, which is what makes it usable on a
camera that mostly turns. While the background is still, the pose is held.

## Pipeline stages (`hawor_pipeline.py`)

| HaWoR stage            | here                                                                 |
|------------------------|----------------------------------------------------------------------|
| detect_track_video     | MediaPipe hand landmarker with ROI search + handedness smoothing     |
| hawor_motion_estimation| MediaPipe metric world landmarks placed in the camera frame by solvePnP |
| hawor_slam             | `telemetry.py` (R_c2w, t_c2w per frame)                               |
| lift to world          | X_w = R_c2w X_c + t_c2w                                              |
| hawor_infiller         | linear interpolation across gaps of up to 12 frames                  |
| MANO mesh              | not available (needs the MANO model + GPU); 21 joints instead        |

To run the real model on a recording, clone HaWoR on a CUDA machine, download
its weights, set `HAWOR_ROOT`, and call `hawor_pipeline.run_official_hawor(path)`.

## Output

`asl_world.jsonl`, one record per frame:

```
{"frame": 120, "t": 4.0,
 "camera": {"source": "visual", "R_c2w": [[...]], "t_c2w": [...], "yaw_pitch_roll_deg": [...],
            "gyro_rad_s": [...], "accel_m_s2": [...], "moving": true, "confidence": 0.93},
 "hands": [{"label": "Right", "confidence": 0.98, "infilled": false, "depth_m": 0.62,
            "reproj_px": 3.1, "R_hand2cam": [[...]], "t_hand_cam": [...],
            "joints_cam": [[x,y,z] x21], "joints_world": [[x,y,z] x21], "wrist_world": [x,y,z]}]}
```

Camera frame is OpenCV's (x right, y down, z forward) of the physical, un-mirrored
image: the live preview is mirrored, but the pipeline flips landmarks back
before any geometry, so world coordinates describe the real scene. World frame
is the camera frame at the first frame (for `insv`, rotated so gravity is +y).
Joints use MediaPipe's 21-point order, in metres. The IMU axis convention from
telemetry-parser has not been verified against OpenCV's on an Ace Pro 2 clip;
see `InsvTelemetry.AXES` in `telemetry.py` if yaw from `insv` and `visual`
disagree in sign on the same recording. Records are written with a 12-frame
delay so the infiller can back-fill a gap before the frames leave the buffer.
`asl_output.json` (the 1 Hz heartbeat) also carries the current camera pose
and each hand's world wrist position.
