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

## Recording training clips (`record_training.py`)

A separate, deliberately minimal recorder for ML training data. It opens the
Insta360, rotates each frame to portrait and writes exactly what the preview
shows to an `.mp4`. Nothing else runs: no telemetry, no hand tracking, no
JSONL stream, no heartbeat, so the clip is a clean record of the signer.

```
python3 record_training.py                 # portrait clips into recordings/training/
python3 record_training.py --label A       # name clips A_<stamp>.mp4
python3 record_training.py --rotate ccw    # camera clamped the other way round
```

Keys: `Space` starts and stops a clip, `a`-`z` sets the label, `m` mirrors the
picture, `q` quits. Settings are locked while a clip is recording so a file is
one geometry end to end. Each clip gets a JSON sidecar
(`<label>_<stamp>.json`) with the label, size, nominal and measured fps,
rotation, mirror state and frame count; its `telemetry` field is always null.
Clips can be fed back through `asl_stream.py --video clip.mp4 --rotate none`
when landmarks are wanted later.

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
| MANO mesh              | via HaMeR (see below) when `--hamer` is on; 21 joints otherwise      |

To run the real model on a recording, clone HaWoR on a CUDA machine, download
its weights, set `HAWOR_ROOT`, and call `hawor_pipeline.run_official_hawor(path)`.

## Hand pose: MediaPipe or HaMeR

MediaPipe is fast and reliable at *finding* hands and following them between
frames, so it always does that job. The hand POSE it reports is a single-view
guess, and it fails exactly where ASL needs it most: a thumb tucked between or
under folded fingers (T, N, M, S) is occluded, so MediaPipe invents it.

`--hamer` swaps the pose stage for HaMeR (Pavlakos et al., CVPR 2024,
https://github.com/geopavlakos/hamer), which fits a full MANO hand with a
learned anatomical prior, so hidden joints are constrained by what a hand can
physically do rather than guessed. Its joints come out in the same 21-point
order MediaPipe uses, so every downstream index stays valid.

```
env/bin/python asl_stream.py --hamer                  # HaMeR on every frame
env/bin/python asl_stream.py --hamer --hamer-every 4  # every 4th frame, MediaPipe between
```

Use `env/bin/python`, not the system python: HaMeR needs PyTorch, which lives
in the project virtual environment. Each record in the JSONL says which model
produced that frame's joints (`pose_model`), so a classifier can train on
HaMeR frames alone.

### Setup

```
python3 -m venv env
env/bin/pip install torch torchvision smplx==0.1.28 yacs einops timm \
    scikit-image pytorch-lightning opencv-python mediapipe scipy \
    "pyobjc-framework-AVFoundation==10.3.2" "numpy<2"
git clone --depth 1 https://github.com/geopavlakos/hamer third_party/hamer
```

Weights go under `third_party/hamer/_DATA/`:
`hamer_ckpts/checkpoints/hamer.ckpt`, `hamer_ckpts/model_config.yaml`,
`data/mano_mean_params.npz` and `data/mano/MANO_RIGHT.pkl`. The official
`fetch_demo_data.sh` pulls a 6 GB bundle; the same files are available
individually from the author's Hugging Face Space, which is a fraction of the
size, and the extra weights in the bundle are a person detector and a body
pose model that MediaPipe replaces here.

**MANO is licensed.** Register at https://mano.is.tue.mpg.de and accept the
terms before using it for anything beyond personal research. `hamer_hands.py`
converts the file from its `chumpy` pickle format to plain numpy on first run,
keeping the original as `MANO_RIGHT.pkl.chumpy.bak`; chumpy is unmaintained and
cannot be installed alongside a current numpy.

### HaWoR

`third_party/HaWoR` holds HaWoR's weights for the egocentric world-space pass
over recordings. Its camera module is DROID-SLAM, which is CUDA-only, so
`telemetry.py` stands in for it on this machine; the hand and infiller networks
run on the Metal backend.

## Body-mounted clips from the Insta360 SD card

The camera strapped to the signer's chest records raw fisheye video with its
own gyro and accelerometer track. Each clip is processed offline:

```
./batch_sdcard.sh VID_20261010_183711_014 VID_20261010_183847_015 ...
```

For every clip this copies it from `/Volumes/Untitled/DCIM/Camera01` into
`recordings/sdcard/`, then runs three steps (landscape clips are rotated
clockwise automatically so the body is at the top of the picture):

```
python3 asl_stream.py --video <clip> --rotate none --fisheye-hfov 78 --hfov 70 --no-preview --thorough --out <stem>_world.jsonl
python3 fill_tracks.py <stem>_world.jsonl <stem>_world_filled.jsonl
python3 render_overlay.py <clip> <stem>_world_filled.jsonl <stem>_overlay.mp4
```

- `--fisheye-hfov` undistorts the raw fisheye before anything looks at it.
- `--thorough` adds full-frame, padded edge and per-hand high-resolution
  detector passes on every frame (about 10 fps offline).
- `fill_tracks.py` gives both hands a record on every frame. Each record's
  `inferred` field says what it is: `observed`, `infilled`, `interpolated`,
  `held` (at most 15 frames after the hand was last seen) or `out_of_view`
  (no pose). A model trained on these must read that field.
- `render_overlay.py` draws each hand exactly where the detector saw it
  (`image_points`), observed in colour and inferred in grey, with the gyro
  pose of the camera on every frame.

Hands that are not the signer's are filtered by physical rules, each measured
on recorded clips (constants and measurements are documented next to them in
`asl_stream.py`): minimum apparent size for a hand within arm's reach; a
learned body anchor for the signer's chin, which a chest camera reads as a
small closed hand; a learned arm direction per hand; distance gating; no new
hand closer than 0.21 m (that is the forearm); and two palms closer than one
palm length are one hand fitted twice.

The IMU axes in the clips are rotated 180 degrees about the vertical relative
to OpenCV's camera frame (measured against visual odometry); `telemetry.py`
corrects for it.

Use a Python with OpenCV and MediaPipe (`/usr/bin/python3` on the development
Mac; the project's `.venv` does not have them).

### Third-party models (not in the repo)

`third_party/` (HaMeR, HaWoR, 6 GB with weights) and `models/` are ignored.
To recreate: clone `geopavlakos/hamer` and `ThunderVVV/HaWoR` into
`third_party/`, download the weights from the Hugging Face Space
`geopavlakos/HaMeR` and the model repo `ThunderVVV/HaWoR`, and on Python 3.9
add `from __future__ import annotations` to `hamer/datasets/utils.py`,
`image_dataset.py` and `json_dataset.py` (they use `X | Y` type unions).
`hamer_hands.py` handles the rest at load time (MANO chumpy conversion, the
doubled `_DATA/` prefix in `model_config.yaml`, OpenGL on macOS, Metal
float32 and contiguity).

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
telemetry-parser was measured against image-based visual odometry on an Ace Pro
2 clip: the IMU frame is turned 180 degrees about the vertical relative to the
lens, which `InsvTelemetry.AXES` corrects. Records are written with a 12-frame
delay so the infiller can back-fill a gap before the frames leave the buffer.
`asl_output.json` (the 1 Hz heartbeat) also carries the current camera pose
and each hand's world wrist position.
