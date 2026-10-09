import os
import sys
import math
import cv2
import json
import time
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Live feed only -- nothing is recorded to disk. The preview window IS the output.
CAMERA_NAME = "insta360"     # which camera to use, matched by name (case-insensitive)
PREVIEW_MAX_WIDTH = 960      # a 1080p window costs ~35ms/frame to draw; 960px costs ~15ms

# Mirror handling. MIRROR_VIEW on means the feed reads like a mirror: the
# signer's right hand appears on the right of the screen. The Insta360 already
# mirrors its own output in Webcam Mode, so an extra cv2.flip here would cancel
# that out and put the right hand back on the left. Press 'm' while running to
# flip it live if the camera's own mirror setting ever changes.
MIRROR_VIEW = True
CAMERA_ALREADY_MIRRORS = True
DETECT_WIDTH = 640           # hand tracking runs on a downscaled copy to stay real-time
STATUS_INTERVAL = 1.0        # how often the JSON heartbeat is refreshed
MODEL_PATH = "hand_landmarker.task"

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
TRACK_MATCH_DIST = 0.18   # how close two wrists must be to count as the same hand
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
        return index, name

    wanted = (requested or CAMERA_NAME).lower()

    if cameras is None:
        print("Error: cannot look up cameras by name -- the AVFoundation binding is missing.")
        print("  Install it with:  python3 -m pip install --user 'pyobjc-framework-AVFoundation==10.3.2'")
        print("  Or pass an explicit index, e.g.  python3 asl_stream.py 0")
        sys.exit(1)

    for index, name in cameras:
        if wanted in name.lower():
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

requested = sys.argv[1] if len(sys.argv) > 1 else None
cam_index, cam_name = resolve_camera(requested)

cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f"Error: could not open '{cam_name}' at index {cam_index}.")
    print("If this is the Insta360, make sure it is in Webcam Mode and reconnect it.")
    sys.exit(1)

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

height, width = frame.shape[:2]
preview_width = min(PREVIEW_MAX_WIDTH, width)
preview_size = (preview_width, int(round(height * preview_width / width)))

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
    )
)


def detect_in(frame, region):
    """Run the landmarker on one sub-region and return landmarks in
    whole-frame normalised coordinates."""
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = (int(region[0] * fw), int(region[1] * fh),
                      int(region[2] * fw), int(region[3] * fh))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    scale = DETECT_WIDTH / crop.shape[1]
    small = cv2.resize(crop, (DETECT_WIDTH, max(1, int(crop.shape[0] * scale))))
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

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
        hands.append((label, score, points))
    return hands


def smooth_handedness(hands):
    """Hold each hand's label steady over time.

    Handedness is decided per frame and flips whenever the model is briefly
    unsure, so each detection is matched to a recent track by wrist position and
    reported as the confidence-weighted majority of that track's history.
    """
    for track in tracks:
        track["idle"] += 1

    out = []
    for label, score, points in hands:
        wrist = points[0]
        match, best = None, TRACK_MATCH_DIST
        for track in tracks:
            distance = math.hypot(track["pos"][0] - wrist[0], track["pos"][1] - wrist[1])
            if distance < best:
                match, best = track, distance
        if match is None:
            match = {"pos": wrist, "votes": deque(maxlen=HANDEDNESS_HISTORY), "idle": 0}
            tracks.append(match)
        match["pos"] = wrist
        match["idle"] = 0
        match["votes"].append((label, max(score, 1e-3)))

        tally = {}
        for voted, weight in match["votes"]:
            tally[voted] = tally.get(voted, 0.0) + weight
        settled = max(tally, key=tally.get)
        out.append((settled, tally[settled] / sum(tally.values()), points))

    tracks[:] = [t for t in tracks if t["idle"] <= TRACK_TTL]
    return out


def roi_around(hands, fw, fh):
    """Padded bounding box (normalised) around the hands found last frame."""
    xs = [x for *_, pts in hands for x, _ in pts]
    ys = [y for *_, pts in hands for _, y in pts]
    pad_x = max(0.12, (max(xs) - min(xs)) * 0.9)
    pad_y = max(0.12, (max(ys) - min(ys)) * 0.9)
    return (max(0.0, min(xs) - pad_x), max(0.0, min(ys) - pad_y),
            min(1.0, max(xs) + pad_x), min(1.0, max(ys) + pad_y))

print(f"Live feed from '{cam_name}' (index {cam_index}) at {width}x{height}.")
print("Hand landmark overlay active. Press 'q' in the preview window to quit.")

frames = 0
start_time = time.time()
last_status = 0.0
last_fps_time = start_time
last_fps_frames = 0
live_fps = 0.0
hands_seen = []
locked_roi = None
misses = 0
search_slot = 0
mirror_view = MIRROR_VIEW
tracks = []

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break

        frames += 1
        now = time.time()

        # Only flip when the camera's own orientation disagrees with the view we
        # want; flipping an already-mirrored feed would undo the mirror.
        if mirror_view != CAMERA_ALREADY_MIRRORS:
            frame = cv2.flip(frame, 1)
        preview = cv2.resize(frame, preview_size)

        # The Insta360's ultra-wide frame defeats the palm detector: a hand that
        # is found instantly inside a focused crop is missed entirely in the full
        # fisheye view. So search one region per frame until a hand turns up, then
        # lock onto it. That keeps the cost at a single detection per frame.
        if locked_roi is not None:
            hands = detect_in(frame, locked_roi)
            if hands:
                misses = 0
            else:
                misses += 1
                if misses > ROI_PATIENCE:
                    locked_roi = None
        else:
            hands = []

        if not hands and locked_roi is None:
            region = SEARCH_REGIONS[search_slot % len(SEARCH_REGIONS)]
            search_slot += 1
            hands = detect_in(frame, region)

        hands = smooth_handedness(hands)

        if hands:
            locked_roi = roi_around(hands, width, height)
            misses = 0

        hands_seen = []
        for label, confidence, points in hands:
            hands_seen.append(label)
            pixels = [(int(x * preview_size[0]), int(y * preview_size[1])) for x, y in points]
            color = HAND_COLORS.get(label, (0, 255, 0))
            for a, b in HAND_CONNECTIONS:
                cv2.line(preview, pixels[a], pixels[b], color, 2)
            for point in pixels:
                cv2.circle(preview, point, 3, (255, 255, 255), -1)
            cv2.putText(preview, f"{label} {confidence:.0%}",
                        (pixels[0][0] - 30, pixels[0][1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if now - last_fps_time >= 0.5:
            live_fps = (frames - last_fps_frames) / (now - last_fps_time)
            last_fps_time = now
            last_fps_frames = frames

        cv2.putText(preview, f"{cam_name} -> Claude Code LIVE  {live_fps:4.1f} fps", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(preview, f"mirror: {'on' if mirror_view else 'off'}  (press 'm' to flip)",
                    (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(preview, f"hands detected: {len(hands_seen)}"
                             + (f"  ({', '.join(hands_seen)})" if hands_seen else ""),
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                    "live_fps": round(live_fps, 1),
                    "hands_detected": len(hands_seen),
                    "handedness": hands_seen,
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
            locked_roi = None          # the tracked box is in the old orientation
            tracks = []                # and the stored labels are for the old one too
            print(f"mirror view {'on' if mirror_view else 'off'}")
except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"Stopped after {frames} frames ({elapsed:.1f}s, {frames / max(elapsed, 1e-6):.1f} fps average).")
