# Portrait video recorder for ML training data, separate from the live pipeline.
#
# asl_stream.py is built around the world-space pipeline: camera telemetry,
# PnP, HaMeR, a JSONL stream per frame. None of that is wanted when the goal
# is simply a clean clip of someone signing. This script opens the Insta360,
# rotates every frame to portrait (the camera is clamped on its side) and
# writes exactly what you see to an .mp4, with a small JSON sidecar that says
# how the clip was recorded. No telemetry, no hand tracking, no heartbeat.
#
#   env/bin/python record_training.py                 # portrait, recordings/training/
#   env/bin/python record_training.py --label A       # clips named A_<stamp>.mp4
#   env/bin/python record_training.py --rotate ccw    # camera clamped the other way
#
# Keys in the preview:  Space  start / stop a clip      a-z  set the label
#                       m      mirror the picture       q    quit
#
# Output, one pair per clip:
#   recordings/training/<label>_<stamp>.mp4    portrait H.264 video, saved as shown (Roboflow-ready)
#   recordings/training/<label>_<stamp>.json   label, size, fps, rotation, frame count
import argparse
import json
import os
import sys
import time

import cv2

CAMERA_NAME = "insta360"     # the only camera this script will run on, matched by name (case-insensitive)
CAMERA_INDEX = 1             # device index the Insta360 gets on this MacBook (built-in camera is 0)
OUT_DIR = os.path.join("recordings", "training")
ROTATION = "cw"              # the Ace Pro 2 is clamped on its side; cw gives an upright portrait frame
ROTATIONS = ["none", "cw", "ccw", "180"]
ROTATE_CODES = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
                "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
PREVIEW_MAX_LONG_EDGE = 720
VIDEO_CODECS = ["avc1", "mp4v"]   # H.264 first: the format Roboflow accepts everywhere


def list_cameras():
    """[(index, name)] of video devices in OpenCV's order, or None without AVFoundation."""
    try:
        import AVFoundation as AVF
    except ImportError:
        return None
    devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    return [(i, str(d.localizedName())) for i, d in enumerate(devices)]


def resolve_camera(requested, allow_any):
    """Camera index to open: explicit index, else the Insta360 found by name."""
    cameras = list_cameras()
    if requested is not None and requested.isdigit():
        index = int(requested)
        name = dict(cameras).get(index, "unknown") if cameras else "unknown"
        if CAMERA_NAME not in name.lower() and not allow_any:
            print(f"Error: index {index} is '{name}', not the {CAMERA_NAME}. Pass --allow-any-camera to use it.")
            sys.exit(1)
        return index, name
    if requested is None and cameras:
        name = dict(cameras).get(CAMERA_INDEX, "")
        if CAMERA_NAME in name.lower():
            return CAMERA_INDEX, name
    if cameras is None:
        print("Error: cannot look up cameras by name -- the AVFoundation binding is missing.")
        print("  Install it with:  python3 -m pip install --user 'pyobjc-framework-AVFoundation==10.3.2'")
        print("  Or pass an explicit index, e.g.  python3 record_training.py 1")
        sys.exit(1)
    wanted = (requested or CAMERA_NAME).lower()
    for index, name in cameras:
        if wanted in name.lower():
            if CAMERA_NAME not in name.lower() and not allow_any:
                print(f"Error: '{name}' is not the {CAMERA_NAME}. Pass --allow-any-camera to use it.")
                sys.exit(1)
            return index, name
    print(f"Error: no camera matching '{wanted}' is connected.")
    for index, name in cameras:
        print(f"  [{index}] {name}")
    print("If this is the Insta360, connect it and put it in Webcam Mode.")
    sys.exit(1)


class ClipRecorder:
    """Writes one clip at a time; frames are saved exactly as given (already
    rotated and mirrored by the caller), so a clip replays as it was seen."""

    def __init__(self, out_dir, fps, rotation, mirrored, camera_name):
        self.out_dir = out_dir
        self.fps = fps
        self.rotation = rotation
        self.mirrored = mirrored
        self.camera_name = camera_name
        self.writer = None
        self.path = None
        self.meta = None
        os.makedirs(out_dir, exist_ok=True)

    @property
    def recording(self):
        return self.writer is not None

    def start(self, label, frame_size):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{label or 'clip'}_{stamp}"
        self.path = os.path.join(self.out_dir, name + ".mp4")
        # H.264 (avc1) in an .mp4 is what Roboflow's browser upload decodes
        # reliably; OpenCV's default MPEG-4 Part 2 (mp4v) often will not play
        # there. mp4v is only a fallback for an OpenCV build without H.264.
        self.codec = None
        for codec in VIDEO_CODECS:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), self.fps, frame_size)
            if self.writer.isOpened():
                self.codec = codec
                break
        else:
            self.writer = None
            raise OSError(f"cannot write {self.path}")
        if self.codec != VIDEO_CODECS[0]:
            print(f"warning: H.264 unavailable, wrote {self.codec}; Roboflow may not accept this file")
        self.size = frame_size
        self.meta = {"label": label, "file": name + ".mp4", "width": frame_size[0], "height": frame_size[1],
                     "fps": self.fps, "rotation": self.rotation, "mirrored": self.mirrored,
                     "camera": self.camera_name, "codec": self.codec, "started": stamp, "frames": 0,
                     "telemetry": None}
        self._t0 = time.time()
        print(f"recording {self.path}")

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = frame[:self.size[1], :self.size[0]]
        self.writer.write(frame)
        self.meta["frames"] += 1

    def stop(self):
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        elapsed = time.time() - self._t0
        self.meta["duration_s"] = round(elapsed, 3)
        self.meta["measured_fps"] = round(self.meta["frames"] / elapsed, 2) if elapsed > 0 else None
        with open(os.path.splitext(self.path)[0] + ".json", "w") as f:
            json.dump(self.meta, f, indent=2)
        print(f"saved {self.path}: {self.meta['frames']} frames, {elapsed:.1f}s")
        return self.path


def rotate(frame, rotation):
    code = ROTATE_CODES[rotation]
    return frame if code is None else cv2.rotate(frame, code)


def draw_label(image, text, origin, color, scale=0.7, thickness=2):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Record portrait training clips from the Insta360, "
                                                 "with no telemetry or hand tracking.")
    parser.add_argument("camera", nargs="?", help="camera index or name fragment (default: the Insta360)")
    parser.add_argument("--label", default=None, help="label used in the clip filenames (a-z keys change it)")
    parser.add_argument("--out-dir", default=OUT_DIR, help="where clips go (default: %(default)s)")
    parser.add_argument("--rotate", default=ROTATION, choices=ROTATIONS,
                        help="rotation applied before saving, for a portrait clip (default: %(default)s)")
    parser.add_argument("--mirror", action="store_true", help="save and show a mirrored picture")
    parser.add_argument("--fps", type=float, default=None, help="ask the camera for this frame rate")
    parser.add_argument("--allow-any-camera", action="store_true", help="use a camera that is not the Insta360")
    args = parser.parse_args()

    cam_index, cam_name = resolve_camera(args.camera, args.allow_any_camera)
    cap = cv2.VideoCapture(cam_index)
    if args.fps and cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print(f"Error: could not open '{cam_name}' at index {cam_index}. Is it in Webcam Mode?")
        sys.exit(1)
    for _ in range(5):
        ok, raw = cap.read()
        if ok:
            break
    else:
        print(f"Error: '{cam_name}' opened but is not delivering frames. Re-enable Webcam Mode and reconnect.")
        cap.release()
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.fps and abs(fps - args.fps) > 0.5:
        print(f"Camera would not accept {args.fps:g} fps; it reports {fps:g} fps.")
    rotation = args.rotate
    mirrored = args.mirror
    label = args.label.upper() if args.label else None
    frame = rotate(raw, rotation)
    h, w = frame.shape[:2]
    scale = min(1.0, PREVIEW_MAX_LONG_EDGE / max(w, h))
    preview_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    recorder = ClipRecorder(args.out_dir, fps, rotation, mirrored, cam_name)
    print(f"{cam_name}: {w}x{h} portrait at {fps:g} fps -> {args.out_dir}")
    print("Space starts/stops a clip, a-z sets the label, m mirrors, q quits.")

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break
            frame = rotate(raw, rotation)
            if mirrored:
                frame = cv2.flip(frame, 1)
            if recorder.recording:
                recorder.write(frame)

            preview = cv2.resize(frame, preview_size)
            if recorder.recording:
                cv2.circle(preview, (24, 28), 9, (0, 0, 255), -1)
                draw_label(preview, f"REC {recorder.meta['frames']}", (42, 36), (0, 0, 255))
            else:
                draw_label(preview, f"label {label or '-'}   space to record", (12, 36), (0, 220, 0))
            cv2.imshow("ASL Training Recorder", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord(' '):
                if recorder.recording:
                    recorder.stop()
                else:
                    # H.264 needs even dimensions; trim a stray odd pixel row/column.
                    recorder.start(label, (frame.shape[1] & ~1, frame.shape[0] & ~1))
            elif recorder.recording:
                continue               # no setting changes mid-clip; a file is one geometry end to end
            elif key == ord('m'):
                mirrored = not mirrored
                recorder.mirrored = mirrored
                print(f"mirror {'on' if mirrored else 'off'}")
            elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                label = chr(key).upper()
                print(f"label {label}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        recorder.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
