import os
import sys
import math
import cv2
import json
import time
import numpy as np
from collections import deque
import argparse
import threading
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from telemetry import intrinsics, make_telemetry
from hawor_pipeline import WorldHandPipeline, hand_center

# Live feed: the preview window is the main output. No video is recorded, but
# per-frame world-space hand records go to asl_world.jsonl and a 1 Hz status
# heartbeat to asl_output.json.
CAMERA_NAME = "insta360"     # the only camera this script will run on, matched by name (case-insensitive)
CAMERA_INDEX = 1             # device index the Insta360 gets on this MacBook (built-in camera is 0)
# Preview cost scales with the pixels drawn, so cap the LONG edge. Capping only
# width meant a portrait (rotated) frame became 960x1707 -- 1.6MP and ~21ms per
# frame to blit, three times the area of the same cap in landscape.
PREVIEW_MAX_LONG_EDGE = 720

# Mirror handling. MIRROR_VIEW on means the feed reads like a mirror: the
# signer's right hand appears on the right of the screen. The Insta360 already
# mirrors its own output in Webcam Mode, so an extra cv2.flip here would cancel
# that out and put the right hand back on the left. Press 'm' while running to
# flip it live if the camera's own mirror setting ever changes.
MIRROR_VIEW = True
CAMERA_ALREADY_MIRRORS = True
# The Ace Pro 2 is clamped on its side for a portrait (upright) view. Frames
# are rotated right after capture, so tracking, telemetry and the world frame
# all see the upright image. Press 'r' while running to cycle orientations.
ROTATION = "cw"              # none | cw | ccw | 180
ROTATIONS = ["none", "cw", "ccw", "180"]
ROTATE_CODES = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
                "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
DETECT_WIDTH = 640           # hand tracking runs on a downscaled copy to stay real-time
STATUS_INTERVAL = 1.0        # how often the JSON heartbeat is refreshed
MODEL_PATH = "hand_landmarker.task"
WORLD_OUTPUT = "asl_world.jsonl"   # per-frame camera telemetry + world-space hand joints
CAMERA_HFOV_DEG = 100.0      # Ace Pro 2 webcam feed, de-warped; used for the pinhole K

# The 21 MediaPipe hand landmarks, grouped into the bones that connect them.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
]
HAND_COLORS = {"Left": (255, 170, 0), "Right": (0, 170, 255)}
# Regions (x0, y0, x1, y1 as fractions of the frame) swept one per frame while
# looking for a hand. The camera sits on the desk looking up, so hands show up
# low and central far more often than anywhere else -- those come first.
SEARCH_REGIONS = [
    (0.15, 0.35, 0.85, 1.00),
    (0.00, 0.25, 0.60, 1.00),
    (0.40, 0.25, 1.00, 1.00),
    (0.25, 0.10, 0.75, 0.75),
    (0.00, 0.00, 1.00, 1.00),
]
ROI_PATIENCE = 8          # frames a lock survives without a detection before re-searching
HANDEDNESS_HISTORY = 12   # frames of handedness history kept per tracked hand
TRACK_MATCH_DIST = 0.18   # how close two hand centres must be to count as the same hand
LOST_HAND_TTL = 45        # frames to keep looking where a hand was last seen before giving up on it
LOST_HAND_PAD = 0.15      # how far around a lost hand's last box to look for it
DETECT_CONFIDENCE = 0.2   # palm detector threshold; fists are low-confidence detections, so keep this low
PRESENCE_CONFIDENCE = 0.2 # landmark model's "there is really a hand here" threshold; fists score low here too
TRACKING_CONFIDENCE = 0.3 # video-mode tracker keeps following the hand above this; low so fists are not dropped
TRACK_CROP_MIN = 0.45     # the tracking crop is at least this fraction of the frame on each side
TRACK_CROP_MARGIN = 0.06  # re-centre the crop when a hand comes this close to its edge
TRACK_DETECT_WIDTH = 768  # tracking crop is fed at this width; fists are small, give the model pixels
TRACK_TTL = 10            # frames a track survives without being seen


def list_cameras():
    """Return [(index, name)] of video capture devices, in the same order OpenCV
    indexes them. Returns None if the AVFoundation binding is unavailable."""
    try:
        import AVFoundation as AVF
    except ImportError:
        return None
    devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    return [(i, str(d.localizedName())) for i, d in enumerate(devices)]


def resolve_camera(requested):
    """Work out which camera index to open.

    Indices shift when devices are plugged, unplugged or wake from sleep, so the
    camera is looked up by name rather than by a hardcoded number. An explicit
    index still wins if one is passed on the command line.
    """
    cameras = list_cameras()

    if requested is not None and requested.isdigit():
        index = int(requested)
        name = dict(cameras).get(index, "unknown") if cameras else "unknown"
        if CAMERA_NAME not in name.lower() and not allow_any_camera:
            print(f"Error: index {index} is '{name}', not the {CAMERA_NAME}. Refusing to use it.")
            print("  Pass --allow-any-camera if you really want this device.")
            sys.exit(1)
        return index, name

    # No argument: try the configured index first, as long as it really is the
    # Insta360. If the indices have shifted, fall through to the name lookup.
    if requested is None and cameras:
        name = dict(cameras).get(CAMERA_INDEX, "")
        if CAMERA_NAME in name.lower():
            return CAMERA_INDEX, name

    wanted = (requested or CAMERA_NAME).lower()

    if cameras is None:
        print("Error: cannot look up cameras by name -- the AVFoundation binding is missing.")
        print("  Install it with:  python3 -m pip install --user 'pyobjc-framework-AVFoundation==10.3.2'")
        print("  Or pass an explicit index, e.g.  python3 asl_stream.py 0")
        sys.exit(1)

    for index, name in cameras:
        if wanted in name.lower():
            if CAMERA_NAME not in name.lower() and not allow_any_camera:
                print(f"Error: '{name}' matched '{wanted}' but is not the {CAMERA_NAME}. Refusing to use it.")
                print("  Pass --allow-any-camera if you really want this device.")
                sys.exit(1)
            return index, name

    print(f"Error: no camera matching '{wanted}' is connected.")
    if cameras:
        print("Cameras currently available:")
        for index, name in cameras:
            print(f"  [{index}] {name}")
    else:
        print("  (no video capture devices found at all)")
    print("If this is the Insta360, connect it and put it in Webcam Mode.")
    sys.exit(1)


if not os.path.exists(MODEL_PATH):
    print(f"Error: missing {MODEL_PATH}.")
    print("Download it with:")
    print("  curl -sSL -o hand_landmarker.task \\")
    print("    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
          "hand_landmarker/float16/1/hand_landmarker.task")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Insta360 ASL stream with camera telemetry and a "
                                             "HaWoR-style world-space hand pipeline.")
parser.add_argument("camera", nargs="?",
                    help=f"camera name fragment or index (default: index {CAMERA_INDEX}, the Insta360)")
parser.add_argument("--video", help="process a recorded file instead of the live camera")
parser.add_argument("--telemetry", default=None,
                    help="camera pose source: static, visual, or insv:<recording>. "
                         "Default: static for live, insv:<file> for --video (falls back to visual)")
parser.add_argument("--hfov", type=float, default=CAMERA_HFOV_DEG, help="horizontal field of view, degrees")
parser.add_argument("--focal", type=float, default=None, help="focal length in pixels (overrides --hfov)")
parser.add_argument("--detect-width", type=int, default=DETECT_WIDTH,
                    help=f"hand detector input width in pixels (default: {DETECT_WIDTH}; smaller is faster)")
parser.add_argument("--out", default=WORLD_OUTPUT, help="world-space JSONL output path")
parser.add_argument("--fps", type=float, default=None,
                    help="request this capture rate from the camera (the Ace Pro 2 tops out at 30 in Webcam Mode)")
parser.add_argument("--rotate", default=ROTATION, choices=ROTATIONS,
                    help="rotate frames after capture for a portrait view (default: %(default)s)")
parser.add_argument("--allow-any-camera", action="store_true",
                    help="permit a camera other than the Insta360 (e.g. the built-in MacBook camera)")
args = parser.parse_args()
if args.detect_width <= 0:
    parser.error("--detect-width must be a positive integer")
allow_any_camera = args.allow_any_camera

if args.video:
    cam_index, cam_name = args.video, os.path.basename(args.video)
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
else:
    cam_index, cam_name = resolve_camera(args.camera)
    cap = cv2.VideoCapture(cam_index)
    video_fps = None
    if args.fps and cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        got = cap.get(cv2.CAP_PROP_FPS)
        if got and abs(got - args.fps) > 0.5:
            print(f"Camera would not accept {args.fps:g} fps; it reports {got:g} fps.")
if not cap.isOpened():
    print(f"Error: could not open '{cam_name}' at index {cam_index}.")
    print("If this is the Insta360, make sure it is in Webcam Mode and reconnect it.")
    sys.exit(1)
# Mirroring is a property of Webcam Mode; a file recorded on the camera is not mirrored.
camera_already_mirrors = CAMERA_ALREADY_MIRRORS and not args.video

# A camera can open but deliver nothing -- an Insta360 that has dropped out of
# Webcam Mode does exactly this, and each failed read costs a one-second
# timeout, so without this check the script just hangs silently.
failures = 0
while failures < 5:
    ok, frame = cap.read()
    if ok:
        break
    failures += 1
else:
    print(f"Error: '{cam_name}' opened but is not delivering frames.")
    print("If this is the Insta360, re-enable Webcam Mode on the camera and reconnect it.")
    cap.release()
    sys.exit(1)

rotation = args.rotate


def rotate(frame):
    code = ROTATE_CODES[rotation]
    return frame if code is None else cv2.rotate(frame, code)


def frame_geometry(frame):
    """(width, height, preview_size, K) for a frame in the current orientation."""
    h, w = frame.shape[:2]
    scale = min(1.0, PREVIEW_MAX_LONG_EDGE / max(w, h))
    preview = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return w, h, preview, intrinsics(w, h, focal_px=args.focal, hfov_deg=args.hfov)


frame = rotate(frame)
width, height, preview_size, K = frame_geometry(frame)

telemetry = None
if args.telemetry is None and args.video:
    try:
        telemetry = make_telemetry(f"insv:{args.video}", K)
    except (SystemExit, Exception) as e:
        print(f"No usable IMU telemetry in {args.video} ({e}); falling back to visual odometry.")
        telemetry = make_telemetry("visual", K)
if telemetry is None:
    telemetry = make_telemetry(args.telemetry or "static", K)
pipeline = WorldHandPipeline(K, telemetry, args.out)

def make_landmarker(mode):
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=DETECT_CONFIDENCE,
            min_hand_presence_confidence=PRESENCE_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
    )


# The palm detector is trained on open palms and gives up on a fist, worst of
# all seen from above. The landmark model has no such trouble once it has a
# hand. So hands are FOUND with a per-image detector on search crops, then
# FOLLOWED with a video-mode landmarker, which carries the landmarks from one
# frame to the next and only asks the palm detector again when it truly
# loses the hand. The video tracker needs a steady picture, so its crop is
# held in place and only moves when a hand nears its edge.
landmarker = make_landmarker(vision.RunningMode.IMAGE)
tracker = make_landmarker(vision.RunningMode.VIDEO)


def detect_in(frame, region, model=None, timestamp_ms=None, detect_width=None):
    """Run the landmarker on one sub-region and return (label, score,
    image landmarks in whole-frame normalised coordinates, metric world
    landmarks in metres centred on the hand)."""
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = (int(region[0] * fw), int(region[1] * fh),
                      int(region[2] * fw), int(region[3] * fh))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    detect_width = detect_width or args.detect_width
    scale = detect_width / crop.shape[1]
    small = cv2.resize(crop, (detect_width, max(1, int(crop.shape[0] * scale))))
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        result = (model or landmarker).detect(image)
    else:
        result = (model or tracker).detect_for_video(image, timestamp_ms)

    hands = []
    for index, marks in enumerate(result.hand_landmarks):
        label, score = "Right", 0.0
        if index < len(result.handedness) and result.handedness[index]:
            category = result.handedness[index][0]
            score = float(category.score)
            # A mirrored right hand is anatomically identical to a left hand, so
            # the model's label is only meaningful relative to image orientation.
            # MediaPipe labels for a mirrored (selfie) view, which is what we
            # show, so its answer is taken as-is here and swapped when the view
            # is NOT mirrored.
            label = category.category_name if mirror_view else (
                "Left" if category.category_name == "Right" else "Right")
        points = [((x0 + p.x * (x1 - x0)) / fw, (y0 + p.y * (y1 - y0)) / fh) for p in marks]
        world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[index]] \
            if index < len(result.hand_world_landmarks) else []
        hands.append((label, score, points, world))
    return hands


def smooth_handedness(hands):
    """Hold each hand's label steady over time.

    Handedness is decided per frame and flips whenever the model is briefly
    unsure, so each detection is matched to a recent track by the centre of the
    back of the hand and reported as the confidence-weighted majority of that
    track's history.
    """
    for track in tracks:
        track["idle"] += 1

    out = []
    for label, score, points, world in hands:
        center = tuple(hand_center(points))
        match, best = None, TRACK_MATCH_DIST
        for track in tracks:
            distance = math.hypot(track["pos"][0] - center[0], track["pos"][1] - center[1])
            if distance < best:
                match, best = track, distance
        if match is None:
            match = {"pos": center, "votes": deque(maxlen=HANDEDNESS_HISTORY), "idle": 0}
            tracks.append(match)
        match["pos"] = center
        match["idle"] = 0
        match["votes"].append((label, max(score, 1e-3)))

        tally = {}
        for voted, weight in match["votes"]:
            tally[voted] = tally.get(voted, 0.0) + weight
        settled = max(tally, key=tally.get)
        out.append((settled, tally[settled] / sum(tally.values()), points, world))

    tracks[:] = [t for t in tracks if t["idle"] <= TRACK_TTL]
    return out


def tracking_crop(hands, previous):
    """A generous, steady crop for the video tracker.

    Big enough to hold both hands with room to move (never below
    TRACK_CROP_MIN of the frame), and only re-centred when a hand gets within
    TRACK_CROP_MARGIN of the current crop's edge. Keeping it still is what
    lets the tracker follow a closed fist from frame to frame.
    """
    xs = [x for _, _, pts, _ in hands for x, _ in pts]
    ys = [y for _, _, pts, _ in hands for _, y in pts]
    if previous is not None:
        x0, y0, x1, y1 = previous
        inside = (min(xs) > x0 + TRACK_CROP_MARGIN or x0 <= 0.0) and (max(xs) < x1 - TRACK_CROP_MARGIN or x1 >= 1.0) \
             and (min(ys) > y0 + TRACK_CROP_MARGIN or y0 <= 0.0) and (max(ys) < y1 - TRACK_CROP_MARGIN or y1 >= 1.0)
        if inside:
            return previous
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max(TRACK_CROP_MIN / 2, (max(xs) - min(xs)) * 0.9)
    half_h = max(TRACK_CROP_MIN / 2 * width / height, (max(ys) - min(ys)) * 0.9)
    return (max(0.0, cx - half_w), max(0.0, cy - half_h), min(1.0, cx + half_w), min(1.0, cy + half_h))


def box_around_points(points, pad):
    xs = [x for x, _ in points]; ys = [y for _, y in points]
    return (max(0.0, min(xs) - pad), max(0.0, min(ys) - pad), min(1.0, max(xs) + pad), min(1.0, max(ys) + pad))


def merge_hands(found, extra):
    """Add detections from a second pass that are not already in `found`."""
    for hand in extra:
        c = hand_center(hand[2])
        if all(math.hypot(*(c - hand_center(h[2]))) > 0.1 for h in found):
            found.append(hand)
    return found


def draw_label(img, text, org, color, scale=0.6, thickness=2):
    """Text with a dark outline so it stays readable over skin and bright backgrounds."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

print(f"Live feed from '{cam_name}' (index {cam_index}) at {width}x{height} (rotation: {rotation}).")
print(f"Camera telemetry: {telemetry.source}. World-space hand records -> {args.out}")
print("Hand landmark overlay active. Press 'q' in the preview window to quit.")

# Hand inference costs ~15ms. Run it on a worker thread so it overlaps the
# camera's ~33ms pacing instead of adding to it: the capture loop then stays
# well inside one frame interval and no longer drops every second frame.
# Landmarks trail the displayed frame by at most one frame.
detect_lock = threading.Lock()
detect_input = {"index": -1, "frame": None}
detect_output = {"index": -1, "hands": [], "roi": None}
detect_stop = threading.Event()
detect_reset = threading.Event()


def detector_worker():
    crop = None              # steady crop the video tracker is following hands in
    lost = {}                # label -> [last box, frames since last seen]
    last_box = {}            # label -> box where that hand was last seen
    misses = 0
    search_slot = 0
    done = -1
    while not detect_stop.is_set():
        with detect_lock:
            index, work = detect_input["index"], detect_input["frame"]
        if work is None or index == done:
            time.sleep(0.001)
            continue
        if detect_reset.is_set():
            detect_reset.clear()
            crop, misses = None, 0
            lost.clear()
            last_box.clear()
            tracks.clear()
        ts_ms = int(index * 1000 / (video_fps or 30))

        # Follow: the video tracker carries landmarks across frames inside a
        # steady crop, which is what keeps a fist from being dropped.
        if crop is not None:
            found = detect_in(work, crop, tracker, ts_ms, TRACK_DETECT_WIDTH)
            if found:
                misses = 0
            else:
                misses += 1
                if misses > ROI_PATIENCE:
                    crop = None
        else:
            found = []

        # Find: no hand being followed, so sweep one search region per frame.
        if not found and crop is None:
            found = detect_in(work, SEARCH_REGIONS[search_slot % len(SEARCH_REGIONS)])
            search_slot += 1

        # One hand followed but the other is missing: spend a second detection
        # looking for it, alternating between where it was last seen and a
        # sweep of the usual regions.
        if found and len(found) < 2:
            region = None
            if lost and search_slot % 2 == 0:
                box, _ = min(lost.values(), key=lambda v: v[1])
                region = (max(0.0, box[0] - LOST_HAND_PAD), max(0.0, box[1] - LOST_HAND_PAD),
                          min(1.0, box[2] + LOST_HAND_PAD), min(1.0, box[3] + LOST_HAND_PAD))
            if region is None:
                region = SEARCH_REGIONS[search_slot % len(SEARCH_REGIONS)]
            search_slot += 1
            found = merge_hands(found, detect_in(work, region))

        found = smooth_handedness(found)
        seen_now = set()
        for label, _, pts, _ in found:
            seen_now.add(label)
            lost.pop(label, None)
            last_box[label] = box_around_points(pts, 0.05)
        for label in list(last_box):
            if label not in seen_now and label not in lost:
                lost[label] = [last_box[label], 0]
        for label in list(lost):
            lost[label][1] += 1
            if lost[label][1] > LOST_HAND_TTL:
                del lost[label]
        if found:
            crop = tracking_crop(found, crop)
            misses = 0

        done = index
        with detect_lock:
            detect_output["index"], detect_output["hands"], detect_output["roi"] = index, found, crop


detector = threading.Thread(target=detector_worker, daemon=True)
detector.start()

frames = 0
start_time = time.time()
last_status = 0.0
last_fps_time = start_time
last_fps_frames = 0
live_fps = 0.0
hands_seen = []
mirror_view = MIRROR_VIEW
tracks = []

try:
    while cap.isOpened():
        ok, raw_frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break

        frame = rotate(raw_frame)
        frames += 1
        now = time.time()
        stream_t = frames / video_fps if video_fps else now - start_time

        # Only flip when the camera's own orientation disagrees with the view we
        # want; flipping an already-mirrored feed would undo the mirror.
        if mirror_view != camera_already_mirrors:
            frame = cv2.flip(frame, 1)
        preview = cv2.resize(frame, preview_size)

        # Hand off this frame and take whatever the detector has finished.
        with detect_lock:
            detect_input["index"], detect_input["frame"] = frames, frame
            hands = list(detect_output["hands"])
            lock_box = detect_output["roi"]

        # HaWoR-style stages: camera telemetry, camera-frame 3D hand, lift to world.
        world_record = pipeline.step(frames, stream_t, frame, hands, width, height, mirrored=mirror_view)
        world_hands = {h["label"]: h for h in world_record["hands"]}
        cam_pose = world_record["camera"]

        if lock_box is not None:
            cv2.rectangle(preview, (int(lock_box[0] * preview_size[0]), int(lock_box[1] * preview_size[1])),
                          (int(lock_box[2] * preview_size[0]), int(lock_box[3] * preview_size[1])),
                          (90, 90, 90), 1)

        hands_seen = []
        for label, confidence, points, _ in hands:
            hands_seen.append(label)
            pixels = [(int(x * preview_size[0]), int(y * preview_size[1])) for x, y in points]
            color = HAND_COLORS.get(label, (0, 255, 0))
            for a, b in HAND_CONNECTIONS:
                cv2.line(preview, pixels[a], pixels[b], color, 2)
            for point in pixels:
                cv2.circle(preview, point, 3, (255, 255, 255), -1)
            # Label below the wrist, clear of the fingers, with an outline so it
            # reads over skin and bright backgrounds.
            lx = min(max(pixels[0][0] - 40, 4), preview_size[0] - 220)
            ly = min(pixels[0][1] + 30, preview_size[1] - 30)
            draw_label(preview, f"{label} {confidence:.0%}", (lx, ly), color, 0.75, 2)
            wh = world_hands.get(label)
            if wh:
                draw_label(preview, f"{wh['depth_m']:.2f} m", (lx, ly + 26), color, 0.6, 2)

        if now - last_fps_time >= 0.5:
            live_fps = (frames - last_fps_frames) / (now - last_fps_time)
            last_fps_time = now
            last_fps_frames = frames

        # No status text on the picture: the fps, mirror/rotation state and
        # camera telemetry all live in asl_output.json instead.
        if now - last_status >= STATUS_INTERVAL:
            last_status = now
            with open("asl_output.json", "w") as f:
                json.dump({
                    "timestamp": now,
                    "status": "live",
                    "camera_active": True,
                    "camera_index": cam_index,
                    "camera_name": cam_name,
                    "resolution": f"{width}x{height}",
                    "mirror_view": mirror_view,
                    "rotation": rotation,
                    "live_fps": round(live_fps, 1),
                    "hands_detected": len(hands_seen),
                    "handedness": hands_seen,
                    "telemetry_source": cam_pose["source"],
                    "camera_pose": cam_pose,
                    "hands_world": {label: {"center_image": h["center_image"], "center_world": h["center_world"],
                                            "depth_m": h["depth_m"]}
                                    for label, h in world_hands.items()},
                    "world_output": args.out,
                    "world_records_written": pipeline.written,
                    "frames": frames,
                    "elapsed_seconds": round(now - start_time, 2),
                }, f, indent=2)

        # Show live camera popup window
        cv2.imshow("Insta360 ASL Stream", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            mirror_view = not mirror_view
            detect_reset.set()         # the tracked box and labels are for the old orientation
            pipeline.reset()           # PnP warm starts and the VO keyframe are too
        if key == ord('r'):
            rotation = ROTATIONS[(ROTATIONS.index(rotation) + 1) % len(ROTATIONS)]
            width, height, preview_size, K = frame_geometry(rotate(raw_frame))
            pipeline.set_intrinsics(K)
            detect_reset.set()
            pipeline.reset()
            print(f"rotation {rotation}: {width}x{height}")
            print(f"mirror view {'on' if mirror_view else 'off'}")
except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    detect_stop.set()
    detector.join(timeout=2)
    pipeline.close()
    landmarker.close()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"Stopped after {frames} frames ({elapsed:.1f}s, {frames / max(elapsed, 1e-6):.1f} fps average).")
